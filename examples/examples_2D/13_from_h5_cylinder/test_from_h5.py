"""Verify FROM_H5 mesh import works correctly.

Tests only the InputManager stage (mesh creation, config reading)
without running the full initialization which requires expensive
JAX compilation.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager
from jaxfluids.domain.mesh_creation.from_h5 import get_cell_centered_levelset
import jax.numpy as jnp
import numpy as np

input_manager = InputManager("cylinder_from_h5.json", "numerical_setup.json")

di = input_manager.domain_information
print("Global number of cells:", di.global_number_of_cells)
print("Domain size:", di.domain_size)
print("Is mesh stretching:", di.is_mesh_stretching)

cc = di.get_global_cell_centers()
cs = di.get_global_cell_sizes()
print("Cell centers x shape:", cc[0].shape,
      "range:", float(jnp.min(cc[0])), "to", float(jnp.max(cc[0])))
print("Cell centers y shape:", cc[1].shape,
      "range:", float(jnp.min(cc[1])), "to", float(jnp.max(cc[1])))
print("Cell sizes x:", cs[0].shape, "dx =", float(jnp.min(cs[0])))
print("Cell sizes y:", cs[1].shape, "dy =", float(jnp.min(cs[1])))

assert di.global_number_of_cells == (160, 100, 1), \
    f"Cell counts mismatch: {di.global_number_of_cells}"
assert abs(di.domain_size[0][0] - (-6.0)) < 1e-10, "X min mismatch"
assert abs(di.domain_size[0][1] - 10.0) < 1e-10, "X max mismatch"
assert abs(di.domain_size[1][0] - (-5.0)) < 1e-10, "Y min mismatch"
assert abs(di.domain_size[1][1] - 5.0) < 1e-10, "Y max mismatch"

# Verify cell centers are within domain
assert float(jnp.min(cc[0])) > -6.0, "Cell centers should be inside domain"
assert float(jnp.max(cc[0])) < 10.0, "Cell centers should be inside domain"

# Verify cell sizes sum to domain length
dx_total = float(jnp.sum(cs[0]))
assert abs(dx_total - 16.0) < 1e-10, f"dx total {dx_total} != 16.0"
dy_total = float(jnp.sum(cs[1]))
assert abs(dy_total - 10.0) < 1e-10, f"dy total {dy_total} != 10.0"

# Verify levelset H5 data is readable via our function
ls = get_cell_centered_levelset("cylinder_mesh.h5")
print("Cell-centered levelset shape:", ls.shape)
assert ls.shape == (160, 100, 1), f"Levelset shape mismatch: {ls.shape}"
assert np.min(ls) < 0, "Levelset should have negative values (inside cylinder)"
assert np.max(ls) > 0, "Levelset should have positive values (outside cylinder)"

# Check SDF values at known points (center should be ~ -0.5)
cx_idx, cy_idx = 60, 50  # approximate center of domain at x=0, y=0
center_val = ls[cx_idx, cy_idx, 0]
print(f"Levelset at center ({cx_idx},{cy_idx}): {center_val:.4f} (expect ~ -0.5)")
assert center_val < 0, "Center of cylinder should be inside (negative)"

print()
print("FROM_H5 feature: ALL CHECKS PASSED")
