"""Run the 3D FROM_H5 cylinder flow example.

Usage:
    python generate_mesh.py       # create H5 mesh
    python run.py                 # run simulation
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager

case_setup = sys.argv[1] if len(sys.argv) > 1 else "cylinder_3d_from_h5.json"

input_manager = InputManager(case_setup, "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)
