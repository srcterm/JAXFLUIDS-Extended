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
    tanh_value: float = None
    ratio_fine_region: float = None
    cells_fine: float = None
    piecewise_parameters: Tuple[PiecewiseStretchingParameters] = None
    file: str = None

class AxisSetup(NamedTuple):
    cells: int
    range: Tuple
    stretching: MeshStretchingSetup

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