import gym
import tensorflow as tf
import numpy as np
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
if __name__=='__main__':
    import robot
else:
    from . import robot

from pudb import set_trace
from matplotlib import pyplot as plt

#'/coppelia_scenes/kuka.ttt
SCENE_FILE = join(dirname(abspath(__file__)), 'coppelia_scenes/kuka.ttt')
print("Scene to be loaded: ")
print(SCENE_FILE)

# Output of gripper.actuate() and gripper.get_amount() need to be scaled to 0 - 1
GRIPPER_RANGE = [0.33, 0.55]  # Measured in simulator
MAX_TORQUES_KUKA7 = [176, 176, 110, 110, 110, 40, 40]  # Taken from Kuka 7 Tech Sheet [Nm]
MAX_VELOCITIES_KUKA7 = [98, 98, 100, 130, 140, 180, 180]  # Taken from Kuka 7 Tech Sheet [deg / s]
MAX_VELOCITIES_KUKA7 = np.array(MAX_VELOCITIES_KUKA7, dtype=np.float32)

MAX_TORQUES_KUKA14 = [320, 320, 176, 176, 110, 40, 40] # Taken from Kuka 14 Tech Sheet [Nm]
MAX_VELOCITIES_KUKA14 = [85, 85, 100, 75, 130, 135, 135] # Taken from Kuka 14 Tech Sheet [deg / s]
MAX_VELOCITIES_KUKA14 = np.array(MAX_VELOCITIES_KUKA14, dtype=np.float32)


class CoppeliaEnv(gym.Env):
    def __init__(self, args, init = False, headless=True):
        # Allows us to restart sim in different force_mode without recreating sim threads
        if not init:
            self._sim = self._start_sim(SCENE_FILE, headless=headless)
        self._prepare_parameters(args)
        self._prepare_robot(args.sub_mock)
        self._prepare_shapes(args.render)
        self._prepare_observation_space()
        self._prepare_action_space()
        self._prepare_subgoal_ranges(args.subgoal_ee_range)

    def step(self, action):
        if self.needs_reset:
            raise Exception('You should reset the environment before you step further.')
        self._apply_action(action)
        self._sim.step()
        observation = self._get_observation()
        done = self._get_done()
        reward = self._get_rew(done)
        info = self._get_info()
        self.timestep += 1
        return observation, reward, done, info

    def reset(self, evalmode=False, hard_reset=False):
        '''Resets the environment to its initial state by setting all the object positions 
        explicitly.
        :param evalmode: If True the target on the table will stay in a specific position.
        :param hard_reset: If True the reset will stop and start the physics simulation. Takes longer but prevents error
        accumulation'''
        if hard_reset:
            print("HARD reset")
            self._sim.stop()
            self._sim.start()
            self._prepare_robot(self._sub_mock)
            self._prepare_shapes(self._render)
        return self._reset(evalmode, random_target=self._random_target)

    def render(self, mode='human'):
        '''gym render function. To render the simulator during simulation, call render(mode='human') once.
        To create rgb pictures, call the function every time you want to render a frame.'''
        if self._gym_cam is None:
            # Add the camera to the scene
            cam_placeholder = Dummy.create()
            cam_placeholder.set_position([0, 0.5, 5])
            cam_placeholder.set_pose([0, 0.5, 5, 1, 0, 0, 0])
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            if mode == "rgb_array":
                self._gym_cam.set_render(RenderMode.OPENGL3)
        if mode == "rgb_array":
            return self._gym_cam.capture_rgb()

    def close(self):
        self._sim.stop()
        self._sim.shutdown()

    def seed(self, seed):
        pass
    
    def set_goal(self, goal):
        if not self._render:
            raise Exception('Do not set goal if you are not rendering. It will not even be present in the simulator.')
        self._meta_goal.set_position(goal, relative_to=None)

    def _start_sim(self, SCENE_FILE, headless):
        sim = PyRep()
        sim.launch(SCENE_FILE, headless=headless)
        # Need sim_timestep set to custom in CoppeliaSim Scene for this method to work.
        sim.set_simulation_timestep(dt=0.03)
        sim.start()
        return sim  

    def _init_step(self):
        '''Need to take a first step in the created simulator so that observations can get created properly.'''
        self._robot.set_joint_target_velocities(np.zeros(shape=[self._max_vel.shape[0],], dtype=np.int32))
        self._sim.step()

    def _prepare_shapes(self, render):
        self._target = Shape('target')
        self._target.set_position(self._target.get_position() + [0, 0.5, 0])
        self._table = Shape('customizableTable')
        self._target_init_pose = self._target.get_pose()
        self._ep_target_pos = self._target.get_position()
        self._gym_cam = None
        self._target.set_renderable(True)
        self._target.set_dynamic(False)
        self._target.set_respondable(False)
        if render:
            print("RENDER")
            from pyrep.const import PrimitiveShape
            self._meta_goal = Shape.create(PrimitiveShape.SPHERE, [0.1,0.1,0.1], renderable=True,
                                           respondable=False)
            self._meta_goal.set_dynamic(False)

    def _prepare_observation_space(self):
        # Do initial step so that observations get generated 
        self._init_step()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._get_observation().shape)

    def _prepare_action_space(self):
        if self.force_mode:
            bound = np.hstack([self._max_torque, 1])
        else:
            bound = np.hstack([self._max_vel, 1])
        self.action_space = gym.spaces.Box(low=-bound, high=bound)
    
    def _prepare_robot(self, sub_mock):
        self._robot = robot.Robot(self.force_mode, self._max_torque, self._max_vel, GRIPPER_RANGE, sub_mock)
        self._init_pos = self._robot.get_joint_positions(gripper_special=False)

    def _prepare_parameters(self, args):
        self._max_episode_steps = args.time_limit
        self._max_vel = MAX_VELOCITIES_KUKA14 * (np.pi / 180)  # API uses rad / s
        self._max_torque = MAX_TORQUES_KUKA14
        self.force_mode = args.force
        self.needs_reset = False
        self._render = args.render
        self._ee_pos = args.ee_pos
        self._ee_j_pos = args.ee_j_pos
        self._sparse_rew = args.sparse_rew
        self._random_target = args.random_target
        self._sub_mock = args.sub_mock
        self.timestep = 0

    def _prepare_subgoal_ranges(self, subgoal_ee_range):
        '''Return the maximal subgoal ranges. In this case:
        [ee_pos, box_pos], which are 2*3 elements. This Method is always
        in flux.'''
        if self._ee_pos:
            # TODO change ee_pos goal range to 1 or smaller or relative goal
            self.subgoal_ranges = np.ones(shape=[3,], dtype=np.float32) * subgoal_ee_range
        elif self._ee_j_pos:
            self.subgoal_ranges = np.ones(shape=[7,], dtype=np.float32) * 3.
        else:
            self.subgoal_ranges = np.ones(shape=[7,], dtype=np.float32) * 3.
        self.target_dim = self._ep_target_pos.shape[0] - 1
        self.subgoal_dim = self.subgoal_ranges.shape[0]

    def _apply_action(self, action):
        ''' Assume action.shape to be [N,] where N-1 is the number of joints in the robotic arm, and action[N]
        indicates the open or close amount of the gripper. This should be changed for 2 arms.
        We treat the gripper as being underactuated. This means that we do not have control over the 2 joints,
        but can only move the two joints together as one control action. cf. Darmstadt Underactuated Sake Gripper Paper.'''
        # Add the 0.01 term because floating point assertion can fail even if action is legitimate. 
        try:
            assert tf.reduce_all(action <= self.action_space.high + 0.01) 
            assert tf.reduce_all(action >= self.action_space.low - 0.01)
        except:
            print("Attention, action_space out of high, low, bounds. Do not generate exception because it sometimes\
                  happens without breaking.")
        if self._sub_mock:
            self._robot.set_joint_target_positions(action[:-1])
            return 
        if not self.force_mode:
            self._robot.set_joint_target_velocities(action[:-1])
        else:
            # In Force Mode, set target_velocities to maximal possible value, then modulate maximal torques 
            signed_vels = np.sign(np.array(action[:-1])) * self._max_vel 
            self._robot.bot[0].set_joint_forces(np.abs(action[:-1]))
            self._robot.set_joint_target_velocities(signed_vels)
        # Gripper not in force mode ever.
        self._robot.actuate(action[-1])

    def _get_rew(self, done):
        if self._sparse_rew:
            if done:
                return 0
            return -1
        return - self._get_distance()
    
    def _get_done(self):
        self.needs_reset = True
        if self._get_distance() < 0.08:
            print("Success")
        elif self.timestep >= self._max_episode_steps - 1:
            pass
        else:
            self.needs_reset = False
        return self.needs_reset

    def _get_distance(self):
        '''Returns L2 distance between arm tip and target.'''
        grip_pos = np.array(self._robot.get_ee_position())
        target_pos = self._ep_target_pos
        return np.linalg.norm(grip_pos - target_pos)

    def _get_observation(self):
        ''' Compute observation. This method is always in flux before we decide on a state
        space.'''
        if self._ee_pos:
            qpos = self._robot.get_ee_position()
            qvel = self._robot.get_ee_velocity()
            observation = np.concatenate([qpos, qvel[0]])
        elif self._ee_j_pos:
            qpos = np.array(np.concatenate([self._robot.get_joint_positions(), self._robot.get_ee_position()]),
                            np.float32)
            qvel = np.array(np.concatenate([self._robot.get_joint_velocities(), self._robot.get_ee_velocity()[0]]),
                            np.float32)
            observation = np.array(np.concatenate([qpos, qvel]), dtype=np.float32)
        else:
            qpos = self._robot.get_joint_positions()
            qvel = self._robot.get_joint_velocities() 
            observation = np.concatenate([qpos, qvel])
        # TODO refactor HIRO code to take obs dict. Nicer to work with
        #observation = {'obs': observation, 'target':self._ep_target_pos}
        return np.array(np.concatenate([observation, self._ep_target_pos[:-1]]), dtype=np.float32)
   
    def _reset_target(self, evalmode, random_target):
        pose = self._target_init_pose
        if random_target and not evalmode:
            pose[:2] = np.random.uniform(low=[-0.08, -0.3], high=[2., 0.95])
        self._target.set_pose(pose)
        return self._target.get_position()

    def _get_info(self):
        return ''
    
    def _reset(self, evalmode, random_target):
        self._robot.set_position(self._init_pos, gripper_special=False)
        self._ep_target_pos = self._reset_target(evalmode, random_target)
        self._sim.step()
        self.needs_reset = False
        self.timestep = 0
        return self._get_observation()

    def _reset_dynamics(self):
        '''Need to reset dynamics because unstable models (like the Darmstadt Sake Gripper) will
        become more and more unstable after repeated position resets.'''
        self._robot.bot[0].reset_dynamic_object()
        self._robot.bot[1].reset_dynamic_object()
        self._target.reset_dynamic_object()
        if self._render:
            self._meta_goal.reset_dynamic_object()

if __name__ == '__main__':

    env = CoppeliaEnv(headless=True, force_mode=False, render=True, ee_pos=True, time_limit=300, sparse_rew=False)
    num_episodes = 0
    x = []
    y = []
    for n in range(1000):
        env.render()
        mock_goal = [0.625 ,-0.01,0.58]
        print("Episode: "+str(num_episodes))
        obs = env.reset()

        env.set_goal(env._target.get_position()+ [0,0,0.1])
        j0 = []
        for t in range(5):
            print("time: {t}")
            if t < 150:
                action = [0, 0,0,0,0,1,0,0]
            else:
                action = [0, 0,0,0,0,-1,0,0]
            action = np.array(action)
            next_obs, rew, done, _  = env.step(action)
            x.append(next_obs[-2])
            y.append(next_obs[-1])
            '''
            print(f"obs_dim: {obs.shape}")
            print(f"target_dim: {env.target_dim}")
            print(f"subgoal_dim: {env.subgoal_dim}")
            print(f"subgoal_ranges: {env.subgoal_ranges}")
            print(obs)
            print(f"distance is {-rew}")
            print(f"target_pos {env._target.get_position()}")
            print(f"meta goal pos {env._meta_goal.get_position()}")'''
            env.set_goal(mock_goal)
            j0.append(obs)
            
            obs = next_obs
    np.save('joints0',j0)
    np.save('posis', np.stack([x, y]))
    env.close()
