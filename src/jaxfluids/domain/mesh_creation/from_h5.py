from typing import Dict, Tuple

import h5py
import numpy as np

from jaxfluids.data_types.case_setup.domain import MeshStretchingSetup


def from_h5(
        axis: int,
        stretching_setup: MeshStretchingSetup,
        **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load non-uniform grid coordinates from an HDF5 file.

    Reads face coordinates for the specified axis from the HDF5 file
    referenced in stretching_setup.file and returns cell centers,
    cell faces, and cell sizes shaped for the JAXFLUIDS tensor-product
    mesh convention.

    :param axis: Spatial axis index (0=x, 1=y, 2=z)
    :param stretching_setup: Stretching configuration containing the
        file path in stretching_setup.file
    :return: cell_centers, cell_faces, cell_sizes as numpy arrays
    """
    coords_keys = ("x_coords", "y_coords", "z_coords")

    with h5py.File(stretching_setup.file, "r") as h5:
        faces = np.array(h5[coords_keys[axis]][:])

    cell_centers = 0.5 * (faces[:-1] + faces[1:])
    cell_sizes = np.diff(faces)

    shape = np.roll(np.s_[-1, 1, 1], axis)
    cell_centers = cell_centers.reshape(shape)
    faces = faces.reshape(shape)
    cell_sizes = cell_sizes.reshape(shape)

    return cell_centers, faces, cell_sizes


def extract_h5_domain_info(file_path: str) -> Dict:
    """Extract domain metadata from an HDF5 mesh file.

    Called during configuration reading to automatically populate
    cell counts and domain ranges when FROM_H5 stretching is used.

    Expected HDF5 layout:
        - Datasets: x_coords, y_coords, z_coords (1D face coordinates)
        - Attribute: non_uniform_grid = True

    :param file_path: Path to the HDF5 mesh file
    :return: Dict with keys cell_counts, domain_ranges, is_mesh_stretching
    """
    metadata = {
        "cell_counts": [0, 0, 0],
        "domain_ranges": [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        "is_mesh_stretching": [False, False, False],
    }

    coords_keys = ("x_coords", "y_coords", "z_coords")

    with h5py.File(file_path, "r") as h5:
        for axis in range(3):
            coords = np.array(h5[coords_keys[axis]][:])
            metadata["cell_counts"][axis] = len(coords) - 1
            metadata["domain_ranges"][axis] = (
                float(coords[0]), float(coords[-1]))
            cell_sizes = np.diff(coords)
            variation = np.abs(np.max(cell_sizes) - np.min(cell_sizes))
            metadata["is_mesh_stretching"][axis] = variation > 1e-12

    return metadata


def get_cell_centered_levelset(file_path: str) -> np.ndarray:
    """Read levelset data from HDF5 and interpolate to cell centers.

    The HDF5 file stores the SDF at grid points (face intersections).
    This function performs trilinear averaging to obtain cell-centered
    values compatible with the JAXFLUIDS levelset buffer layout.

    :param file_path: Path to the HDF5 mesh file containing a
        'levelset' dataset
    :return: Cell-centered levelset array of shape (nx, ny, nz)
    """
    with h5py.File(file_path, "r") as h5:
        sdf = np.array(h5["levelset"][:])

    return (
        sdf[:-1, :-1, :-1] + sdf[1:, :-1, :-1] +
        sdf[:-1, 1:, :-1]  + sdf[:-1, :-1, 1:]  +
        sdf[1:, 1:, :-1]   + sdf[1:, :-1, 1:]   +
        sdf[:-1, 1:, 1:]   + sdf[1:, 1:, 1:]
    ) / 8.0
