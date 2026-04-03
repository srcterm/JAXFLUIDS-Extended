"""Verify FROM_H5 mesh import for both stretched and uniform meshes.

Tests the InputManager stage (mesh creation, config reading) without
running the full simulation.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax.numpy as jnp
import numpy as np
from jaxfluids import InputManager
from jaxfluids.domain.mesh_creation.from_h5 import get_cell_centered_levelset


def check_mesh(config_file, h5_file, expected_cells, expected_domain,
               expect_stretching, label):
    """Verify a single FROM_H5 mesh configuration."""
    print(f"\n--- {label} ---")
    print(f"  Config: {config_file}")
    print(f"  Mesh:   {h5_file}")

    im = InputManager(config_file, "numerical_setup.json")
    di = im.domain_information

    print(f"  Cells:      {di.global_number_of_cells}")
    print(f"  Domain:     {di.domain_size}")
    print(f"  Stretching: {di.is_mesh_stretching}")

    # Cell counts
    assert di.global_number_of_cells == expected_cells, \
        f"Cell counts: {di.global_number_of_cells} != {expected_cells}"

    # Domain ranges
    for axis in range(2):
        lo, hi = expected_domain[axis]
        assert abs(di.domain_size[axis][0] - lo) < 1e-10, \
            f"Axis {axis} min: {di.domain_size[axis][0]} != {lo}"
        assert abs(di.domain_size[axis][1] - hi) < 1e-10, \
            f"Axis {axis} max: {di.domain_size[axis][1]} != {hi}"

    # Mesh stretching flags
    for axis in range(3):
        assert di.is_mesh_stretching[axis] == expect_stretching[axis], \
            f"Axis {axis} stretching: {di.is_mesh_stretching[axis]} != {expect_stretching[axis]}"

    # Cell centers within domain
    cc = di.get_global_cell_centers()
    assert float(jnp.min(cc[0])) > expected_domain[0][0]
    assert float(jnp.max(cc[0])) < expected_domain[0][1]
    assert float(jnp.min(cc[1])) > expected_domain[1][0]
    assert float(jnp.max(cc[1])) < expected_domain[1][1]

    # Cell sizes sum to domain length
    cs = di.get_global_cell_sizes()
    dx_total = float(jnp.sum(cs[0]))
    dy_total = float(jnp.sum(cs[1]))
    x_len = expected_domain[0][1] - expected_domain[0][0]
    y_len = expected_domain[1][1] - expected_domain[1][0]
    assert abs(dx_total - x_len) < 1e-10, f"dx sum {dx_total} != {x_len}"
    assert abs(dy_total - y_len) < 1e-10, f"dy sum {dy_total} != {y_len}"

    # Levelset
    ls = get_cell_centered_levelset(h5_file)
    nx, ny = expected_cells[0], expected_cells[1]
    assert ls.shape == (nx, ny, 1), f"Levelset shape: {ls.shape}"
    assert np.min(ls) < 0, "Levelset should have negative values (inside)"
    assert np.max(ls) > 0, "Levelset should have positive values (outside)"

    print(f"  PASSED")


if __name__ == "__main__":
    domain = [(-10.0, 12.0), (-10.0, 10.0)]

    check_mesh(
        config_file="cylinder_from_h5_stretched.json",
        h5_file="cylinder_stretched.h5",
        expected_cells=(500, 400, 1),
        expected_domain=domain,
        expect_stretching=(True, True, False),
        label="Stretched (PIECEWISE-extracted)")

    # FROM_H5 always reports stretching=True for active axes since the
    # stretching type is set, regardless of whether the grid is uniform.
    check_mesh(
        config_file="cylinder_from_h5_uniform.json",
        h5_file="cylinder_uniform.h5",
        expected_cells=(440, 400, 1),
        expected_domain=domain,
        expect_stretching=(True, True, False),
        label="Uniform")

    print("\n=== ALL CHECKS PASSED ===")
