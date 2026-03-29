"""Run the FROM_H5 cylinder flow example.

Usage:
    python generate_mesh.py   # creates cylinder_mesh.h5
    python run.py             # runs the simulation
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from jaxfluids import InputManager, InitializationManager, SimulationManager

# SETUP SIMULATION
input_manager = InputManager("cylinder_from_h5.json", "numerical_setup.json")
initialization_manager = InitializationManager(input_manager)
sim_manager = SimulationManager(input_manager)

# RUN SIMULATION
jxf_buffers = initialization_manager.initialization()
sim_manager.simulate(jxf_buffers)
