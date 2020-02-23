"""
A Franka Panda reaches for 10 randomly places targets.
This script contains examples of:
    - Linear (IK) paths.
    - Scene manipulation (creating an object and moving it).
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820 as Kuka
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from robot import EZGripper


import numpy as np
import math

LOOPS = 10
SCENE_FILE = join(dirname(abspath(__file__)), 'coppelia_scenes/kuka.ttt')
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.set_simulation_timestep(dt=0.10)
pr.start()
agent = Kuka()
agent.set_control_loop_enabled(True)
agent.set_motor_locked_at_zero_velocity(True)
agent.set_joint_forces(np.ones([7,]) * 800)
# We could have made this target in the scene, but lets create one dynamically
target = Shape('target')
target.set_respondable(False)
target.set_dynamic(False)

position_min, position_max = [0.8, -0.2, 2.0], [1.0, 0.2, 2.0]

starting_joint_positions = agent.get_joint_positions()

pos = target.get_position()
for i in range(LOOPS):

    # Reset the arm at the start of each 'episode'
    agent.set_joint_positions(starting_joint_positions)

    # Get a random position within a cuboid and set the target position
    #pos = list(np.random.uniform(position_min, position_max))
    pos[0] += 0.1
    target.set_position(pos)

    # Get a path to the target (rotate so z points down)
    try:
        path = agent.get_path(
            position=pos, euler=[0, math.radians(180), 0])
    except ConfigurationPathError as e:
        print('Could not find path')
        continue

    # Step the simulation and advance the agent along the path
    done = False
    while not done:
        done = path.step()
        pr.step()

    print('Reached target %d!' % i)

pr.stop()
pr.shutdown()
