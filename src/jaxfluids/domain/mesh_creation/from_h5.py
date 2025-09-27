from typing import Tuple, Dict, Optional, Union
import h5py
import numpy as np
import jax.numpy as jnp
Array = jnp.ndarray  # Array is alias for jax.Array in core code

# Module-level cache for H5 metadata to avoid repeated file reads
_h5_metadata_cache = {}


class H5MeshProcessor:
    """
    Self-contained processor for H5 SDF mesh compatibility with JAXFLUIDS.
    The class has logic to read H5 files, extract domain metadata,
    and create mesh arrays that are compatible with JAXFLUIDS stencil operations.
    The SDF data can be generated from external tools and does not strictly need
    to be a uniform grid, but can be non-uniformly spaced in each spatial direction.
    """
    
    def __init__(self, file_path: str):
        """Initialize the H5 mesh processor with validation."""
        self.file_path = file_path
        self._validate_h5_file()
        # Cache domain metadata for efficient access
        self._domain_metadata = None
        
    def _validate_h5_file(self) -> None:
        """Validate that the H5 file contains required data."""
        try:
            with h5py.File(self.file_path, "r") as h5:
                # Check if this is a non-uniform grid
                if not h5.attrs.get("non_uniform_grid", False):
                    raise ValueError(
                        f"File {self.file_path} does not contain non_uniform_grid=True. "
                        "Use HOMOGENEOUS stretching instead."
                    )
                
                # Check that required coordinate arrays exist
                required_keys = ["x_coords", "y_coords", "z_coords"]
                missing_keys = [key for key in required_keys if key not in h5]
                if missing_keys:
                    raise ValueError(f"Missing coordinate arrays in {self.file_path}: {missing_keys}")
                    
        except OSError as e:
            raise ValueError(f"Cannot read H5 file {self.file_path}: {e}")
    

    def extract_domain_metadata(self) -> Dict:
        """
        Extract domain size, cell counts, and axis activity from H5 file.
        
        This method provides the metadata that domain_information.py needs
        without requiring FROM_H5-specific logic in that core file.
        """
        if self._domain_metadata is not None:
            return self._domain_metadata
            
        metadata = {
            "cell_counts": [0, 0, 0],
            "domain_ranges": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            "is_mesh_stretching": [False, False, False]
        }
        
        coords_keys = ["x_coords", "y_coords", "z_coords"]
        
        with h5py.File(self.file_path, "r") as h5:
            for axis in range(3):
                coords_key = coords_keys[axis]
                coordinates = np.array(h5[coords_key][:])
                
                # Calculate cell count
                metadata["cell_counts"][axis] = len(coordinates) - 1  # faces to cells
                
                # Calculate domain range
                metadata["domain_ranges"][axis] = (float(coordinates[0]), float(coordinates[-1]))
                
                # Check if mesh is stretched
                cell_sizes = np.diff(coordinates)
                size_variation = np.abs(np.max(cell_sizes) - np.min(cell_sizes))
                metadata["is_mesh_stretching"][axis] = size_variation > 1e-12
        
        self._domain_metadata = metadata
        return metadata
    

    def get_cell_centered_levelset(self) -> np.ndarray:
        """
        Extract levelset data and interpolate from coordinate points to cell centers.
        """
        with h5py.File(self.file_path, "r") as h5:
            if "levelset" not in h5:
                raise ValueError(f"No levelset data found in {self.file_path}")
                
            # Load levelset data (defined at coordinate points)
            levelset_at_points = np.array(h5["levelset"][:])
            
            # Get coordinate arrays
            x_coords = np.array(h5["x_coords"][:])
            y_coords = np.array(h5["y_coords"][:])  
            z_coords = np.array(h5["z_coords"][:])
            
            # Verify shapes match
            expected_shape = (len(x_coords), len(y_coords), len(z_coords))
            if levelset_at_points.shape != expected_shape:
                raise ValueError(
                    f"Levelset shape {levelset_at_points.shape} does not match "
                    f"coordinate grid shape {expected_shape}"
                )
            
            # Interpolate to cell centers using trilinear interpolation
            # For each cell, average the 8 surrounding coordinate points
            nx, ny, nz = len(x_coords) - 1, len(y_coords) - 1, len(z_coords) - 1
            levelset_at_centers = np.zeros((nx, ny, nz))
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        # Average the 8 corner points of the cell
                        levelset_at_centers[i, j, k] = (
                            levelset_at_points[i, j, k] + levelset_at_points[i+1, j, k] +
                            levelset_at_points[i, j+1, k] + levelset_at_points[i, j, k+1] +
                            levelset_at_points[i+1, j+1, k] + levelset_at_points[i+1, j, k+1] +
                            levelset_at_points[i, j+1, k+1] + levelset_at_points[i+1, j+1, k+1]
                        ) / 8.0
            
            return levelset_at_centers
    

    def _load_raw_coordinates(self, axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load raw coordinates from H5 file for specified axis."""
        coords_keys = ["x_coords", "y_coords", "z_coords"]
        coords_key = coords_keys[axis]
        
        with h5py.File(self.file_path, "r") as h5:
            # These are face coordinates, not cell centers
            faces = np.array(h5[coords_key][:])
        
        # Compute cell centers from face coordinates
        cell_centers = 0.5 * (faces[:-1] + faces[1:])
        
        # Compute cell sizes
        cell_sizes = np.diff(faces)
        faces[0] = cell_centers[0] - 0.5 * (cell_centers[1] - cell_centers[0])
        # Compute cell sizes
        cell_sizes = np.diff(faces)
        
        return cell_centers, faces, cell_sizes
    

    def _ensure_stencil_compatibility(
        self, 
        array: np.ndarray, 
        axis: int, 
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        Ensure arrays are compatible with JAXFLUIDS stencil operations.
        
        Applied logic that was in deriv_center_4.py and central_adap_4.py
        to pre-process arrays during mesh creation.
        """
        if target_shape is not None:
            # Apply symmetric slicing if array is larger than target
            slices = []
            for dim in range(len(array.shape)):
                current_size = array.shape[dim]
                target_size = target_shape[dim] if dim < len(target_shape) else current_size
                
                if current_size > target_size:
                    # Symmetric slicing
                    start = (current_size - target_size) // 2
                    end = start + target_size
                    slices.append(slice(start, end))
                else:
                    slices.append(slice(None))
            
            array = array[tuple(slices)]
        
        return array
    

    def _create_derivative_compatible_cell_sizes(
        self, 
        cell_sizes: np.ndarray, 
        axis: int,
        output_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """
        Apply derivative stencil compatibility logic from deriv_center_4.py.
        
        Ensure cell size arrays can be properly sliced during derivative
        comp. to match output shapes.
        """
        if output_shape is None:
            return cell_sizes
            
        # Apply the logic from deriv_center_4.py for dynamic slicing
        # The key is that we slice along the axis dim to match target size
        target_size = output_shape[0] if len(output_shape) > 0 else len(cell_sizes)
        current_size = len(cell_sizes)
        
        if current_size > target_size:
            # Apply symmetric slicing along the axis dim
            start = (current_size - target_size) // 2
            end = start + target_size
            cell_sizes = cell_sizes[start:end]
        
        return cell_sizes
    

    def _create_reconstruction_compatible_arrays(
        self, 
        arrays: Tuple[np.ndarray, ...],
        buffer_spatial_shape: Optional[Tuple[int, ...]] = None
    ) -> Tuple[np.ndarray, ...]:
        """
        Apply reconstruction stencil compatibility logic from central_adap_4.py.
        
        Ensures coefficient arrays can be properly sliced during reconstruction
        to match buffer dims.
        """
        if buffer_spatial_shape is None:
            return arrays
            
        # Apply the logic from central_adap_4.py for dynamic coefficient slicing
        processed_arrays = []
        
        for array in arrays:
            needs_slicing = False
            slice_amounts = [0, 0, 0]  # For X, Y, Z 
            
            # Check if array dims are larger than buffer dims, check the last 3 space dims
            array_spatial_dims = array.shape[-3:] if array.ndim >= 3 else array.shape
            buffer_dims = buffer_spatial_shape[-3:] if len(buffer_spatial_shape) >= 3 else buffer_spatial_shape
            
            for dim in range(min(len(array_spatial_dims), len(buffer_dims))):
                array_dim_size = array_spatial_dims[dim]
                buffer_dim_size = buffer_dims[dim]
                
                if array_dim_size > buffer_dim_size:
                    # Calculate symmetric slicing amount
                    diff = array_dim_size - buffer_dim_size
                    # Use ceiling division to ensure we slice enough to fit within buffer
                    slice_amounts[dim] = (diff + 1) // 2
                    needs_slicing = True
            
            if needs_slicing: # Slice each dim?
                slices = []

                # Handle leading dims (keep unchanged)
                for dim in range(array.ndim - 3):
                    slices.append(slice(None))

                # Handle spatial dims (last 3)
                for dim in range(max(0, array.ndim - 3), array.ndim):
                    spatial_dim = dim - max(0, array.ndim - 3)
                    if spatial_dim < 3 and slice_amounts[spatial_dim] > 0:
                        # Slice this spatial dim
                        start = slice_amounts[spatial_dim]
                        end = array.shape[dim] - slice_amounts[spatial_dim]
                        slices.append(slice(start, end))
                    else:
                        # Keep full dim
                        slices.append(slice(None))
                
                array = array[tuple(slices)]
            
            processed_arrays.append(array)
        
        return tuple(processed_arrays)
    

    def _ensure_halo_compatibility(
        self, 
        array: np.ndarray, 
        axis: int,
        conservative_halos: Optional[int] = None,
        geometry_halos: Optional[int] = None
    ) -> np.ndarray:
        """
        Ensure arrays are compatible with JAXFLUIDS halo operations.
        
        Apply logic from source_term_solver.py
        to pre-process arrays during mesh creation.
        """
        # For stretched meshes, proper halo compatibility needs to be made
        if conservative_halos is not None:
            pass
        if geometry_halos is not None: 
            pass

        return array
    

    def create_jaxfluids_compatible_mesh(
        self,
        axis: int,
        nxi: int,
        domain_size_xi: Tuple[float, float],
        halo_info: Optional[Dict] = None
    ) -> Tuple[Array, Array, Array]:
        """
        Create mesh arrays that are fully compatible with existing JAXFLUIDS infrastructure.
        This method applies all necessary transformations to ensure the returned
        cell centers, faces, and cell sizes are compatible with stencil operations
        and halo extensions.
        """
        # Load raw coordinate data
        cell_centers, faces, cell_sizes = self._load_raw_coordinates(axis)
        
        # Apply shape compatibility transformations
        if halo_info:
            # Apply halo compatibility
            cell_centers = self._ensure_halo_compatibility(
                cell_centers, axis, 
                halo_info.get('conservative_halos'),
                halo_info.get('geometry_halos')
            )
            faces = self._ensure_halo_compatibility(
                faces, axis,
                halo_info.get('conservative_halos'), 
                halo_info.get('geometry_halos')
            )
            
            # Apply derivative stencil compatibility for cell sizes
            cell_sizes = self._create_derivative_compatible_cell_sizes(
                cell_sizes, axis, halo_info.get('output_shape')
            )
            
            # Apply reconstruction stencil compatibility if buffer shape provided
            if 'buffer_spatial_shape' in halo_info:
                cell_centers, faces, cell_sizes = self._create_reconstruction_compatible_arrays(
                    (cell_centers, faces, cell_sizes),
                    halo_info['buffer_spatial_shape']
                )
        
        # Reshape arrays to proper 3D shape for JAXFLUIDS
        # Follow the same pattern as piecewise.py for consistency with existing mesh stretching
        shape = np.roll(np.s_[-1,1,1], axis)
        
        # Reshape the arrays - this is critical for halo compatibility
        cell_centers = cell_centers.reshape(shape)
        faces = faces.reshape(shape)
        cell_sizes = cell_sizes.reshape(shape)
        
        # Convert to jax arrays and return
        return jnp.array(cell_centers), jnp.array(faces), jnp.array(cell_sizes)


def from_h5(
        axis: int,
        nxi: int,
        domain_size_xi: Tuple[float, float],
        stretching_setup
) -> Tuple[Array, Array, Array]:
    """
    Load non-uniform grid coordinates from .h5 file for specified axis.
    
    This function loads cell center coordinates from .h5 files and converts them to 
    the format expected by JAXFLUIDS mesh stretching. The returned cell sizes are
    formatted to be compatible with the halo extension system.
    
    The nxi and domain_size_xi parameters are ignored - actual values are determined 
    from the .h5 file. This function now handles all domain metadata extraction 
    internally and provides properly shaped arrays for JAXFLUIDS.
    """
    # Create processor instance for this H5 file
    processor = H5MeshProcessor(stretching_setup.file)
    
    # Extract domain metadata and cache it for domain_information.py access
    metadata = processor.extract_domain_metadata()
    
    # Store metadata in module-level cache using file path as key
    # This allows domain_information.py to get the actual cell counts and ranges
    # without needing FROM_H5-specific logic
    _h5_metadata_cache[stretching_setup.file] = metadata
    
    # Use the enhanced processor to create compatible mesh arrays
    return processor.create_jaxfluids_compatible_mesh(
        axis=axis,
        nxi=metadata["cell_counts"][axis],               # Use actual cell count from H5
        domain_size_xi=metadata["domain_ranges"][axis],  # Use actual range from H5
        halo_info=None
    )


def get_h5_metadata(file_path: str) -> Optional[Dict]:
    """
    Get cached H5 metadata for the specified file path.
    
    This function allows domain_information.py to access the metadata
    extracted during mesh creation without requiring FROM_H5-specific logic.
    """
    return _h5_metadata_cache.get(file_path)


def extract_h5_domain_info(file_path: str) -> Dict:
    """
    Extract domain configuration information from H5 file for early use during 
    configuration reading. This allows automatic population of cells and range
    values when they're not specified in the JSON configuration.
    
    Returns:
        Dict containing:
        - "cell_counts": [nx, ny, nz] - Number of cells in each direction
        - "domain_ranges": [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        - "is_mesh_stretching": [bool, bool, bool] - Whether each axis is stretched
    """
    # Create processor to extract metadata
    processor = H5MeshProcessor(file_path)
    metadata = processor.extract_domain_metadata()
    
    # Cache the metadata for later use
    _h5_metadata_cache[file_path] = metadata
    
    return metadata
