import numpy as np
import coppeliagym
from pudb import set_trace




env = coppeliagym.CoppeliaEnv(headless=True, force_mode=False)
pos = env._robot.bot[0].get_joint_positions()
pos[1] = 0.8
pos[0] = -1.5
env.render()
env._robot.bot[0].set_joint_positions(pos)
env._sim.step()
distance = np.linalg.norm(np.array(env._robot.bot[0].get_tip().get_position()) - np.array(env._target.get_position()))
set_trace()

print(distance)
for i in range(100):
    print(i)
    env._sim.step()

env.close()

