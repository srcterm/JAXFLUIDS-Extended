"""Generate HDF5 mesh files for the FROM_H5 cylinder validation.

Creates two meshes matching the 05_cylinder_flow reference case
(500x400, domain [-10,12]x[-10,10]):

1. cylinder_stretched.h5 - PIECEWISE grid extracted from JAXFLUIDS
2. cylinder_uniform.h5   - Uniform grid, same domain and cell count

Both include the cylinder SDF (r=0.5 at origin) at face vertices.
"""

import os
import sys

import numpy as np
import h5py


def cylinder_sdf(x, y, z, radius=0.5):
    """Signed distance function for a cylinder (2D circle in x-y)."""
    return np.sqrt(x**2 + y**2) - radius


def write_h5(filename, x_faces, y_faces, z_faces):
    """Write mesh and levelset to HDF5."""
    xp, yp, zp = np.meshgrid(x_faces, y_faces, z_faces, indexing="ij")
    sdf = cylinder_sdf(xp, yp, zp)

    with h5py.File(filename, "w") as h5:
        h5.attrs["non_uniform_grid"] = True
        h5.create_dataset("x_coords", data=x_faces)
        h5.create_dataset("y_coords", data=y_faces)
        h5.create_dataset("z_coords", data=z_faces)
        h5.create_dataset("levelset", data=sdf)

    nx = len(x_faces) - 1
    ny = len(y_faces) - 1
    print(f"  {filename}: {nx}x{ny}x1, "
          f"[{x_faces[0]:.1f},{x_faces[-1]:.1f}]x"
          f"[{y_faces[0]:.1f},{y_faces[-1]:.1f}]")


def generate_stretched_mesh():
    """Extract the PIECEWISE grid from the reference 05_cylinder_flow config."""
    ref_dir = os.path.join(os.path.dirname(__file__),
                           "..", "05_cylinder_flow")
    ref_case = os.path.join(ref_dir, "cylinder_flow.json")
    ref_num = os.path.join(ref_dir, "numerical_setup.json")

    from jaxfluids import InputManager
    im = InputManager(ref_case, ref_num)
    di = im.domain_information

    faces = di.get_global_cell_faces()
    x_faces = np.squeeze(np.array(faces[0]))
    y_faces = np.squeeze(np.array(faces[1]))
    z_faces = np.array([0.0, 1.0])

    write_h5("cylinder_stretched.h5", x_faces, y_faces, z_faces)


def generate_uniform_mesh():
    """Create a uniform grid with same domain as reference.

    Uses 440x400 cells so that dx = dy = 0.05 (unity aspect ratio),
    required by the levelset solver.
    """
    x_faces = np.linspace(-10.0, 12.0, 441)  # 440 cells, dx = 0.05
    y_faces = np.linspace(-10.0, 10.0, 401)  # 400 cells, dy = 0.05
    z_faces = np.array([0.0, 1.0])

    write_h5("cylinder_uniform.h5", x_faces, y_faces, z_faces)


if __name__ == "__main__":
    print("Generating meshes...")
    generate_stretched_mesh()
    generate_uniform_mesh()
    print("Done.")
