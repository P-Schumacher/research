import numpy as np
from pyrep.robots.arms.lbr_iiwa_14_r820 import LBRIwaa14R820 as Kuka 
from pyrep.robots.end_effectors.gripper import Gripper
from pyrep.const import RenderMode




class EZGripper(Gripper):
    def __init__(self, count=0):
        string = 'EZGripper_Gen2'
        super().__init__(count,string,  ['Left_Base', 'Right_Base'])


class Robot:
    def __init__(self, force_mode, max_torque, max_velocity, gripper_range, sub_mock):
        '''Robot class that hold arm and gripper.
        Gripper dimension: 2 joints
        Arm dimension: 7 joints'''
        self._max_torque = max_torque
        self._max_velocity = max_velocity
        self._gripper_range = gripper_range
        self.bot = [Kuka(), EZGripper()]
        self.bot = self._set_vel_ctrl(self.bot, force_mode, sub_mock)
        self._num_joints = np.sum([x._num_joints for x in self.bot])   
        self.arm_joints = self.bot[0]._num_joints

    def set_position(self, pos, gripper_special=False):
        ''' Set the arm position directly without considering physics.'''
        if gripper_special:
            self.bot[0].set_joint_positions(pos)
        else:
            self.bot[0].set_joint_positions(pos[:-2])
            self.bot[1].set_joint_positions(pos[-2:])

    def set_joint_target_positions(self, target, gripper_special):
        self.bot[0].set_joint_target_positions(target[:self._num_joints])

    def get_joint_positions(self, gripper_special=True):
        if gripper_special:
            return self.bot[0].get_joint_positions()
        else:
            return np.hstack([x.get_joint_positions() for x in self.bot])
    
    def get_joint_velocities(self, gripper_special=True):
        if gripper_special:
            return self.bot[0].get_joint_velocities()
        else:
            return np.hstack([x.get_joint_velocities() for x in self.bot])

    def get_joint_forces(self, gripper_special):
        '''Returns the forces as measured by the joints. Not to mistake
        with set_joint_forces() which limits the maximum amount of force a particular 
        joint can exert to reach the set target_velocity/position etc.'''
        if gripper_special:
            return self.bot[0].get_joint_forces()
        else:
            return np.hstack([x.get_joint_forces() for x in self.bot])

    def get_orientation(self):
        pass

    def set_joint_target_velocities(self, target):
        '''Set target velocities of arm and gripper joints, or just the arm. If just arm_joints, you can call
        actuate() to move the gripper.'''
        assert target.shape[0] == self._num_joints or target.shape[0] == self.arm_joints
        self.bot[0].set_joint_target_velocities(target[:self.arm_joints])
        if target.shape[0] == self._num_joints:
            self.bot[1].set_joint_target_velocities(target[self.arm_joints:])
 
    def get_ee_position(self):
        '''Returns the cartesian position of the arm tip.'''
        return self.bot[0].get_tip().get_position()

    def get_ee_velocity(self):
        return self.bot[0].get_tip().get_velocity()

    def actuate(self, grip_action):
        '''Applies an opening or closing force on the gripper and return a number [0,1]
        that indicates how much it is closed. We recompute that number as the EZGripper
        is not calibrated right.
        :params: grip_action: Number between -1 and 1
        grip_action > 0 opens the gripper.
        grip_action < 0 closes the gripper.
        The value of grip_action determines the velocity of the movement.'''
        vel = np.abs(grip_action) * 10
        if np.sign(grip_action) < 0:
            direction  = 0
        else:
            direction = 1
        amount = self.get_open_amount()
        # Don't open gripper further if it's already parallel.
        if not (np.mean(amount) > 0 or direction == 1):
            vel = 0
        _ = self.bot[1].actuate(direction, vel)
        return amount 
    
    def get_open_amount(self):
        '''Return the open-amount of the gripper. The EZ Gripper is opened at
        0.55 and closed at 0.33, we need to scale this to 0 and 1.'''
        amount = np.hstack(self.bot[1].get_open_amount())
        return (amount - self._gripper_range[0]) / (self._gripper_range[1] - self._gripper_range[0])

    def _set_vel_ctrl(self, robot, force_mode, sub_mock):
        '''Disables the PID controllers which allows target_velocity control, which is more natural for RL.
        We also lock the motor at zero_velocity such that the arm doesn't fall down immediately.
        Then we set the maximum allowed force of every joint to be the ones given by the Kuka Technical Sheets.'''
        for part in robot:
            # Control loop enabled with sub_mock 
            part.set_control_loop_enabled(sub_mock)
            part.set_motor_locked_at_zero_velocity(False)
            if not force_mode:
                forces = [45, 45] # Random values, not clear from Sake Gripper Tech Sheet
                if part._num_joints == len(self._max_torque):
                    forces = self._max_torque
                part.set_joint_forces(forces)
                part.set_motor_locked_at_zero_velocity(True)
        return robot
    

