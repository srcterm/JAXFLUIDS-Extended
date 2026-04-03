"""Run a cylinder flow case.

Usage:
    python generate_mesh.py                            # create H5 meshes
    python run.py cylinder_reference.json              # PIECEWISE reference
    python run.py cylinder_from_h5_stretched.json      # FROM_H5 stretched
    python run.py cylinder_from_h5_uniform.json        # FROM_H5 uniform
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager

case_setup = sys.argv[1] if len(sys.argv) > 1 else "cylinder_from_h5_stretched.json"

input_manager = InputManager(case_setup, "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)
