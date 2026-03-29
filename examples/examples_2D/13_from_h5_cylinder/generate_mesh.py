"""Generate an HDF5 mesh file with a cylinder levelset
for testing the FROM_H5 mesh import feature.

Creates a uniform 2D grid with matching cell sizes in x
and y (required for levelset unity aspect ratio), plus the
SDF for a cylinder at the origin. The grid and levelset are
stored together in a single HDF5 file that JAXFLUIDS can
import via the FROM_H5 stretching type.
"""

import numpy as np
import h5py


def cylinder_sdf(x, y, z, cx=0.0, cy=0.0, radius=0.5):
    """Signed distance function for a cylinder (2D circle in x-y)."""
    return np.sqrt((x - cx)**2 + (y - cy)**2) - radius


def main():
    dx = 0.1
    x_min, x_max = -6.0, 10.0
    y_min, y_max = -5.0, 5.0

    nx = int(round((x_max - x_min) / dx))
    ny = int(round((y_max - y_min) / dx))

    x_faces = np.linspace(x_min, x_max, nx + 1)
    y_faces = np.linspace(y_min, y_max, ny + 1)
    z_faces = np.array([0.0, 1.0])

    xp, yp, zp = np.meshgrid(x_faces, y_faces, z_faces, indexing="ij")
    sdf = cylinder_sdf(xp, yp, zp, cx=0.0, cy=0.0, radius=0.5)

    output_path = "cylinder_mesh.h5"
    with h5py.File(output_path, "w") as h5:
        h5.attrs["non_uniform_grid"] = True
        h5.create_dataset("x_coords", data=x_faces)
        h5.create_dataset("y_coords", data=y_faces)
        h5.create_dataset("z_coords", data=z_faces)
        h5.create_dataset("levelset", data=sdf)

    print(f"Wrote {output_path}")
    print(f"  Grid: {nx} x {ny} x 1  (dx = {dx})")
    print(f"  Domain: [{x_min}, {x_max}] x [{y_min}, {y_max}] x [0, 1]")


if __name__ == "__main__":
    main()
