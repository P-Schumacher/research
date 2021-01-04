import gym
from pudb import set_trace
import tensorflow as tf
import numpy as np
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.const import RenderMode
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from utils.math_fns import huber, euclid
from . import robot

COUNTER = 20 # How many iterations should the boxes be untouchable after a wrong sequence

class CoppeliaEnv(gym.Env):
    def __init__(self, cnf, init=False):
        # Allows us to restart sim in different force_mode without recreating sim threads
        if not init:
            self._sim = self._start_sim(**cnf.sim)
        self._prepare_parameters(**cnf.params)
        self._prepare_robot(self._sub_mock, self._gripper_range)
        self._prepare_shapes(self._render, self._flat_agent)
        self._prepare_observation_space()
        self._prepare_action_space()
        self._prepare_subgoal_ranges(**cnf.subgoals)
        if cnf.params.ee_pos and cnf.params.ee_j_pos:
            raise Exception(f'Decide on your state space! ee_pos:{cnf.params.ee_pos} ee_j_pos:{cnf.params.ee_j_pos}')

    def step(self, action, reward_fn):
        if self._needs_reset:
            raise Exception('You should reset the environment before you step further.')
        self._apply_action(action)
        self._sim.step()
        reward = self._get_rew(action, self._task)
        done = self._get_done()
        observation = self._get_observation()
        info = self._get_info()
        self._timestep += 1
        self._total_it +=1 
        if reward_fn:
            self._reversal = True
        else:
            self._reversal = False
        if self._render and self._double_buttons:
            if self._state_b1 == 1:
                self._target.set_color([1, 0, 1])
            if self._state_b2 == 1:
                self._target2.set_color([1, 0, 1])
        return observation, reward, done, info

    def reset(self, evalmode=False):
        '''Resets the environment to its initial state by setting all the object positions 
        explicitly.
        Configuration trees are used as an efficient way of recreating the robot on episode
        reset if a part breaks during simulation.
        :param evalmode: If True the target on the table will stay in a specific position.'''
        # This resets the gripper to its initial state, even if it broke during table touches
        self._sim.set_configuration_tree(self._initial_arm_conftree)
        self._sim.set_configuration_tree(self._initial_gripper_conftree)
        if self._render and self._double_buttons:
            self._reset_button_colors()
        state = self._reset(evalmode)
        # Control flow for task success
        self.success = False
        self.mega_reward = True
        self._state_b1 = 0
        self._state_b2 = [0 if self._double_buttons else 1][0]
        self._stop_counter = COUNTER
        return state

    def render(self, mode='human'):
        '''gym render function. To render the simulator during simulation, call render(mode='human') once.
        To create rgb pictures, call the function every time you want to render a frame.'''
        if self._gym_cam is None:
            # Add the camera to the scene
            cam_placeholder = Dummy.create()
            cam_placeholder.set_pose([0, -0.5, 5, 1, 0, 0, 0])
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam2 = VisionSensor('Vision_sensor')
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            self._gym_cam2.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            if mode == "rgb_array":
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)
        if mode == "rgb_array":
            return self._gym_cam.capture_rgb()

    def close(self):
        '''Shuts simulator completely down'''
        self._sim.stop()
        self._sim.shutdown()

    def seed(self, seed):
        '''Would change the seed of the RNGs but this simulator is a physics simulator, so there should be no RNGs.'''
        pass
    
    def set_goal(self, goal):
        '''Set a visual sphere in the environment to visualize the goal of the meta-agent. Only works if *_render* is true.'''
        if not self._render:
            raise Exception('Do not set goal if you are not rendering. It will not even be present in the simulator.')
        self._meta_goal.set_position(goal, relative_to=None)

    def _start_sim(self, scene_file, render_scene_file, headless, sim_timestep, render):
        '''Starts the simulation. We use different scene files for training and rendering, as unused objects still 
        slow down the simulation.
        :param sim_timestep: The timestep between actionable frames. The physics simulation has a timestep of 5ms,
        but the control commands are repeated until the *sim_timestep* is reached'''
        sim = PyRep()
        scene_file = [scene_file if not render else render_scene_file][0]
        scene_file = join(dirname(abspath(__file__)), scene_file)
        sim.launch(scene_file, headless=headless)
        # Need sim_timestep set to custom in CoppeliaSim Scene for this method to work.
        sim.set_simulation_timestep(dt=sim_timestep)
        sim.start()
        return sim  

    def _init_step(self):
        '''Need to take a first step in the created simulator so that observations can get created properly.'''
        self._robot.set_joint_target_velocities(np.zeros(shape=[self._max_vel.shape[0],], dtype=np.int32))
        self._sim.step()

    def _prepare_shapes(self, render, flat_agent):
        '''Create python handles for the objects in the scene.'''
        self._table = Shape('customizableTable')
        self._target = Shape('target')
        self._target.set_color([0.84,0.15,0.16])
        self._pos_b1 = self._target.get_position()
        self._target_init_pose = self._target.get_pose()
        if self._double_buttons:
            self._target2 = Shape('target1')
            self._pos_b2 = self._target2.get_position()
            self._target_init_pose2 = self._target2.get_pose()
        self._gym_cam = None
        self._target.set_renderable(True)
        self._target.set_dynamic(False)
        self._target.set_respondable(False)
        if self._double_buttons:
            self._target2.set_renderable(True)
            self._target2.set_dynamic(False)
            self._target2.set_respondable(False)
        if render and not flat_agent and not self._init:
            self._init = True
            print("RENDER")
            from pyrep.const import PrimitiveShape
            self._meta_goal = Shape.create(PrimitiveShape.SPHERE, [0.1,0.1,0.1], renderable=True,
                                           respondable=False, color=[0.22,0.34,0.14]) #color=[0,0.05,1])
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
    
    def _prepare_robot(self, sub_mock, gripper_range):
        self._robot = robot.Robot(self.force_mode, self._max_torque, self._max_vel, gripper_range, sub_mock)
        self._init_pos = self._robot.get_joint_positions(gripper_special=False)
        self._initial_arm_conftree = self._robot.bot[0].get_configuration_tree()
        self._initial_gripper_conftree = self._robot.bot[1].get_configuration_tree()

    def _prepare_parameters(self, **kwargs):
        # Float or integer params
        kwargs = {f'_{key}': value for key, value in kwargs.items()}
        kwargs['force_mode'] = kwargs.pop('_force')
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        # Params that need special types
        self._max_vel = np.array(self._max_vel, np.float64) * (np.pi / 180)  # API uses rad / s
        self._max_torque = np.array(self._max_torque, np.float64)
        self._distance_fn = self._get_distance_fn(self._distance_function)
        self.time_limit = [600 if 'two' in self._task else 300][0]
        self._state_max = np.array(self._state_max, np.float32)
        self._state_min = np.array(self._state_min, np.float32)
        # Control flow
        self._timestep = 0
        self._needs_reset = True
        self._init = False
        self._total_it = 0
        self._reversal = False
        self._double_buttons = bool('two' in self._task)
        # Need these before reset to get observation.shape
        self._state_b1 = 0
        self._state_b2 = 0
        self._init_gripper = [6.499e-1, -6.276e-1, 1.782]
        # For state normalization
        self._state_mean = np.load('./environments/state_mean_two_button.npy')
        self._state_std = np.load('./environments/state_sigma_two_button.npy')

    def _prepare_subgoal_ranges(self, ee_goal, j_goal, ej_goal):
        '''Generate subgoal ranges that the HIRO uses to determine its subgoal dimensions.  Not useful
        for other algorithms.'''
        if self._spherical_coord:
            self.subgoal_ranges = [1 for x in range(3)]
        elif  self._ee_pos:
            self.subgoal_ranges = [ee_goal for x in range(3)]
        elif self._ee_j_pos:
            self.subgoal_ranges = [ej_goal[1] for x in range(ej_goal[0])]
        else:
            self.subgoal_ranges = [j_goal for x in range(7)]
        if not self._double_buttons:
            self.target_dim = self._pos_b1.shape[0] - 1
        else:
            #self.target_dim = (self._ep_target_pos.shape[0] - 1) * 2 + 2 + 1
            self.target_dim = (self._pos_b1.shape[0] - 1) * 2 + 2 
        self.subgoal_dim = len(self.subgoal_ranges)

    def _apply_action(self, action):
        ''' Assume action.shape to be [N,] where N-1 is the number of joints in the robotic arm, and action[N]
        indicates the open or close amount of the gripper. This should be changed for 2 arms.
        We treat the gripper as being underactuated. This means that we do not have control over the 2 joints,
        but can only move the two joints together as one control action. cf. Darmstadt Underactuated Sake Gripper Paper.'''
        try:
            assert tf.reduce_all(action <= self.action_space.high + 0.0001) 
            assert tf.reduce_all(action >= self.action_space.low - 0.0001)
        except:
            print("Attention, action_space out of high, low, bounds. Your robot probably broke.")
        if self._sub_mock:
            # Use a simulated perfect sub-agent with PID controllers
            self._robot.set_joint_target_positions(action[:-1])
            return 
        if not self.force_mode:
            # velocity control mode
            self._robot.set_joint_target_velocities(action[:-1])
        else:
            # In Force Mode, set target_velocities to maximal possible value, then modulate maximal torques 
            signed_vels = np.sign(np.array(action[:-1])) * self._max_vel 
            self._robot.bot[0].set_joint_forces(np.abs(action[:-1]))
            self._robot.set_joint_target_velocities(signed_vels)
        # Gripper is handled with underactuation
        self._robot.actuate(action[-1])

    def _get_rew(self, action, task):
        if task == 'dense':
            return self._get_rew_dense(action)

        if task == 'sparse_one_button':
            return self._get_rew_sparse_one_button(action)

        if task == 'dense_two_button':
            if not self._reversal:
                self._state_b1 = False
                self._state_b2 = True
            else:
                self._state_b1 = True
                self._state_b2 = False

            return self._get_rew_dense_two_button(action)

        if task == 'sparse_two_button':
            if not self._reversal:
                self._state_b1 = False
                self._state_b2 = True
            else:
                self._state_b1 = True
                self._state_b2 = False

            return self._get_rew_sparse_two_button(action)

        if task == 'sparse_two_button_sequential':
            return self._get_rew_sparse_two_button_sequential_reset_counter_wrapper(action)

        raise Exception('''Pick one of the valid reward types:
                        1) dense 
                        2) sparse_one_button
                        3) dense_one_button NOPE
                        4) sparse_two_button
                        5) sparse_two_button_sequential''')

    def _get_rew_dense(self, action):
        '''A dense reward based on the distance between the end-effector
        and the box is given. A success is recorded if 
        d < *self._touch_distance*'''
        dist = - self._get_distance(self._pos_b1)
        if tf.abs(dist) < self._touch_distance:
            self._state_b1 = True
        return  dist - self._action_regularizer * tf.square(tf.norm(action))

    def _get_rew_sparse_one_button(self, action):
        '''A sparse reward is given if the distance between the end-effector
        and the box is below *self._touch_distance*'''
        rew = -1.
        # One button touch task
        if self._get_distance(self._pos_b1) < self._touch_distance:
            rew += 1
            self._state_b1 = True
        return rew

    def _get_rew_dense_two_button(self, action):
        '''Compute a sparse reward based on two boxes where you just have
        to make contact with the right one. There is a negative reward
        for the incorrect contact and the episode only ends after correct 
        contact.'''
        if not self._reversal:
            rew = - self._get_distance(self._pos_b1)
            if self._get_distance(self._pos_b2) < self._touch_distance:
                rew -= 1000
        else:
            rew = - self._get_distance(self._pos_b2)
            if self._get_distance(self._pos_b1) < self._touch_distance:
                rew -= 1000
        if self._distance_query_switcher(0) < self._touch_distance:
            if not self._reversal:
                self._state_b1 = 1
            else:
                self._state_b2 = 1
            #print('button 1 pressed')
        if self._get_distance(self._pos_b2)  < self._touch_distance:
            pass
            #print('button 2 pressed')
        return rew

    def _get_rew_sparse_two_button(self, action):
        '''Compute a sparse reward based on two boxes where you just have
        to make contact with the right one. There is a negative reward
        for the incorrect contact and the episode only ends after correct 
        contact.'''
        rew = -1.
        if self._distance_query_switcher(0) < self._touch_distance and not self._state_b1 == 1:
            self._state_b1 = 1
            print('button 1 pressed')
            rew += 1
        if self._get_distance(self._pos_b2)  < self._touch_distance:
            print('button 2 pressed')
            rew -= -1
        return rew

    def _get_rew_sparse_two_button_sequential_reset_counter_wrapper(self, action):
        '''If *self._stop_counter* is enabled, this wrapper blocks button
        touches and resulting rewards for *COUNTER* steps after every failure.'''
        if self._stop_counter >= COUNTER:
            rew, punishment = self._get_rew_sparse_two_button_sequential(action)
            if punishment and self._reset_on_wrong_sequence:
                self._reset_counter_settings()
        if self._reset_on_wrong_sequence:
            self._stop_counter += 1
        return rew

    def _reset_counter_settings(self):
        '''Resets the appropriate settings if *self._reset_on_wrong_sequence* is 
        enabled and an incorrect touching was detected.'''
        self._state_b1 = False
        self._state_b2 = False
        self.mega_reward = True
        self._stop_counter = 0
        if self._render:
            self._reset_button_colors()

    def _get_rew_sparse_two_button_sequential(self, action):
        '''A sparse reward is given if box 1 is touched before box 2.
        The success condition evaluates to *True* no matter what order the 
        boxes were touched in. This ensures the validity of the MDP.'''
        punishment = False
        rew = -1.
        if self._distance_query_switcher(0) < self._touch_distance and not self._state_b1 == 1:
            self._state_b1 = 1
            print('button 1 pressed')
        if self._distance_query_switcher(1) < self._touch_distance and not self._state_b2 == 1:
            self._state_b2 = 1
            print('button 2 pressed')
        if self._state_b2 == 1 and not self._state_b1 == 1:
            self.mega_reward = False
        if (self._state_b1 == 1 and self._state_b2 == 1) and self.mega_reward:
            rew += 50
            print('MEGA reward')
        if (self._state_b1 == 1 and self._state_b2 == 1) and not self.mega_reward:
            print('FAILURE Punishment')
            rew -= self._punishment
            punishment = True
        return rew, punishment

    def _distance_query_switcher(self, box):
        '''Switches the boxes used for distance and reward computation
        if the *self._reversal* signal is given.'''
        if not self._reversal:
            if box == 0:
                return self._get_distance(self._pos_b1)
            else:
                return self._get_distance(self._pos_b2)
        else:
            if box == 0:
                return self._get_distance(self._pos_b2)
            else:
                return self._get_distance(self._pos_b1)
    
    def _get_done(self):
        '''Compute a *done* which is returned and a *success* variable, which is internally saved.
        A proper MDP should distinguish between successfull episodes where a certain condition
        was met (i.e. button touched) and episodes which ended because of a timelimit. We use
        *done* to indicate that an episode should be reset and *success* to indicate if 
        the value of a terminal state should be set to zero. Alternatively, one could
        add the timestep to the state. cf. Time Limits in Reinforcement Learning, Pardo et al.'''
        self._needs_reset = True
        if self._state_b1 == 1 and self._state_b2 == 1:
            print("Success")
            self.success = True
        elif self._timestep >= self.time_limit - 1:
            pass
        else:
            self._needs_reset = False
        return self._needs_reset

    def _get_distance(self, target_pos):
        grip_pos = np.array(self._robot.get_ee_position(), dtype=np.float32)
        return self._distance_fn(grip_pos, target_pos)
    
    def _get_distance_fn(self, func_string):
        if func_string == 'euclid':
            return self._euclid_distance
        elif func_string == 'huber':
            return self._huber_distance
        else:
            raise Exception('Non valid distance measure specified. Allowed are: *euclid* and *huber*')

    def _euclid_distance(self, grip_pos, target_pos):
        '''Returns L2 distance between arm tip and target.'''
        return euclid(grip_pos - target_pos)

    def _huber_distance(self, grip_pos, target_pos, delta=1.):
        'Returns distance between arm tip and target using the Huber distance function.'
        dist = tf.constant(grip_pos - target_pos, dtype=tf.float32)
        return huber(dist, delta)

    def _get_observation(self):
        ''' Compute observation. 
        *ee_j_pos* means end-effector and joint positions and velocities are known.
        The *else* branch corresponds to the case that only the joint positions and 
        velocities are known.
        The returned observation is either [robot_state, box_position]
        or in the double button case
        [robot_state, box1_position, box1_active_status, box2_position, box1_active_status]'''
        if self._ee_j_pos:
            qpos = np.concatenate([self._robot.get_ee_position(), self._robot.get_joint_positions()])
            qvel = np.concatenate([self._robot.get_ee_velocity()[0], self._robot.get_joint_velocities()])
            #observation = np.array(np.concatenate([qpos, qvel, np.array([self._timestep])]), dtype=np.float32)
            observation = np.array(np.concatenate([qpos, qvel]), dtype=np.float32)
        else:
            qpos = self._robot.get_joint_positions()
            qvel = self._robot.get_joint_velocities() 
            observation = np.concatenate([qpos, qvel])
        if not self._double_buttons:
            target = self._pos_b1[:-1]
        else:
            target = np.concatenate([self._pos_b1[:-1], [self._state_b1], self._pos_b2[:-1],
                                     [self._state_b2]], axis=0)
            #target = np.concatenate([self._pos_b1[:-1], self._pos_b2[:-1]], axis=0)
        ret = np.array(np.concatenate([observation, target]), dtype=np.float32)
        if self._normalize:
            #ret = self._normalizer(ret)
            ret = self._linear_scaler(ret)
        return ret

    def _normalizer(self, x):
        return (x - self._state_mean) / (self._state_std)

    def _linear_scaler(self, x):
        return (x - self._state_min) / (0.5*(self._state_max - self._state_min))-1


   
    def _reset_target(self, evalmode):
        pose = self._target_init_pose.copy()
        if self._double_buttons:
            pose2 = self._target_init_pose2.copy()
        if self._random_target and not evalmode or self._random_eval_target and evalmode:
            x, y = self._sample_in_circular_reach()
            pose[:2] = [x, y]
            self._target.set_pose(pose)
            if self._double_buttons:
                self._set_double_target(pose, pose2)
                self._target2.set_pose(pose2)
        else:
            self._target.set_pose(pose)
            if self._double_buttons:
                self._target2.set_pose(pose2)
        if not self._double_buttons:
            return np.array(self._target.get_position(), dtype=np.float32), None
        else:
            return np.array(self._target.get_position(), dtype=np.float32), np.array(self._target2.get_position(),
                                                                                        dtype=np.float32)
    def _set_double_target(self, pose, pose2):
        iterations = 0
        while True:
            iterations += 1
            x, y = self._sample_in_circular_reach()
            pose2[:2] = [x, y]
            if euclid(pose[:2] - pose2[:2]) >= self._minimum_dist:
                      break
            if iterations >= 10000:
                print('Limit exceeded for target setup. Look at *_reset_target()*')
                break

    def _reset_button_colors(self):
        self._target.set_color([0, 0, 1])
        self._target2.set_color([1, 0, 0])

    def _get_info(self):
        return ''
    
    def _reset(self, evalmode):
        ''' Reset target velocities BEFORE positions. As arm and gripper are reset independently 
         in *set_position()*
         there is an additional simulation timestep between them. If the velocities are not reset properly, the
         reset position of the robot will drift. This drift is small for the arm but can cause the gripper to
         explode and destabilize the simulation.'''
        self._robot.set_joint_target_velocities(np.zeros(shape=self._init_pos.shape))
        self._robot.set_position(self._init_pos)
        self._pos_b1, self._pos_b2 = self._reset_target(evalmode)
        self._sim.step()
        self._needs_reset = False
        self._timestep = 0
        return self._get_observation()
    
    def _sample_in_circular_reach(self):
        '''Uses polar coordinates to sample in a circular region around the arm. Radius was measured
        by letting the arm flail around for some time and then drawing a line from the two recorded points
        that were furthest apart.'''
        r = np.random.uniform(0.3, 1.867/2 - 0.1)
        theta = np.random.uniform(-np.pi, np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x + 0.622, y - 0.605 
    
