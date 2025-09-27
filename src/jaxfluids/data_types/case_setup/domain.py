from typing import NamedTuple, Tuple

import jax.numpy as jnp

# DOMAIN & MESH
class DomainDecompositionSetup(NamedTuple):
    split_x: int = 1
    split_y: int = 1
    split_z: int = 1

class PiecewiseStretchingParameters(NamedTuple):
    type: str
    cells: int
    upper_bound: float
    lower_bound: float

class MeshStretchingSetup(NamedTuple):
    type: str
    tanh_value: float = 0.0
    ratio_fine_region: float = 0.0
    cells_fine: float = 0.0
    piecewise_parameters: Tuple[PiecewiseStretchingParameters] = ()
    file: str = None  # Path to .h5 for FROM_H5

class AxisSetup(NamedTuple):
    cells: int = 1
    range: Tuple = (0.0, 1.0)
    stretching: MeshStretchingSetup = None

class DomainSetup(NamedTuple):
    x: AxisSetup
    y: AxisSetup
    z: AxisSetup
    decomposition: DomainDecompositionSetup
    active_axes: Tuple[str]
    active_axes_indices: Tuple[int]
    inactive_axes: Tuple[str]
    inactive_axes_indices: Tuple[int]
    dim: int