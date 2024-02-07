import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pydantic import BaseModel
import logging
from omegaconf import OmegaConf

from pyvortex.vortex_interface import VortexInterface, AppMode
from peg_in_hole.settings import app_settings

logger = logging.getLogger(__name__)
robot_logger = logging.getLogger('robot_state')

""" Names in vortex scene """

""" 
TODO
- Add logger
- Robot settings to yaml
- render option
- Update function
    - Get sim time
    - Send command
    - Read outputs
    - Log everything in json file
- Vortex log
    - Deactivate or specify path
"""

""" Vortex Scene Inputs, Outputs and Parameters"""


class VX_Inputs(BaseModel):
    j2_vel_id: str = 'j2_vel_id'
    j4_vel_id: str = 'j4_vel_id'
    j6_vel_id: str = 'j6_vel_id'


class VX_Outputs(BaseModel):
    hand_pos_rot: str = 'hand_pos_rot'
    j2_pos_real: str = 'j2_pos_real'
    j4_pos_real: str = 'j4_pos_real'
    j6_pos_real: str = 'j6_pos_real'
    j2_vel_real: str = 'j2_vel_real'
    j4_vel_real: str = 'j4_vel_real'
    j6_vel_real: str = 'j6_vel_real'
    j2_torque: str = 'j2_torque'
    j4_torque: str = 'j4_torque'
    j6_torque: str = 'j6_torque'
    socket_force: str = 'socket_force'
    socket_torque: str = 'socket_torque'
    plug_force: str = 'plug_force'
    plug_torque: str = 'plug_torque'


VX_IN = VX_Inputs()
VX_OUT = VX_Outputs()


""" Kinova Robot Interface """


class RPL_Insert_3DoF(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None, task_cfg=None):
        init_start_time = time.time()

        """Load config"""
        self._get_robot_config()

        # General params
        self.task_cfg = task_cfg

        self.sim_time = 0.0
        self.step_count = 0  # Step counter

        self._init_obs_space()
        self._init_action_space()

        """ RL Hyperparameters """
        # Actions
        self.action_coeff = self.task_cfg.rl.hparams.action_coeff

        # Reward
        self.reward_min_threshold = self.task_cfg.rl.reward.reward_min_threshold
        self.min_height_threshold = self.task_cfg.rl.reward.min_height_threshold
        self.reward_weight = self.task_cfg.rl.reward.reward_weight

        """ Sim Hyperparameters """
        # TODO: To YAML and pydantic data class
        # Vortex
        self.h = 1.0 / 100.0  # Simulation time step
        self.t_init_step = 5.0  # Time to move arm to the insertion position
        self.t_pause = 1.0  # Pause time
        self.t_pre_insert = 5.0  # used to be 0.8 for 7DOF
        self.t_insertion = 2.5
        self.init_steps = int(self.t_init_step / self.h)  # Initialization steps
        self.pause_steps = int(self.t_pause / self.h)  # Pause time step after one phase
        self.pre_insert_steps = int(self.t_pre_insert / self.h)  # go up to the insertion phase
        self.insertion_steps = int(self.t_insertion / self.h)  # Insertion time (steps)
        self.max_insertion_steps = 2.0 * self.insertion_steps  # Maximum allowed time to insert
        self.xpos_hole = 0.529  # x position of the hole
        self.ypos_hole = -0.007  # y position of the hole
        self.max_misalign = 2.0  # maximum misalignment of joint 7
        self.min_misalign = -2.0  # minimum misalignment of joint 7
        self.insertion_misalign = (np.pi / 180.0) * (
            np.random.uniform(self.min_misalign, self.max_misalign)
        )  # joint 6 misalignment, negative and positive
        # self.insertion_misalign = (np.pi/180.0) * self.max_misalign    # joint 6 misalignment, set max
        self.pre_insertz = 0.44  # z distance to be covered in pre-insertion phase
        self.insertz = 0.07  # z distance to be covered in insertion phase, though may change with actions

        """ Robot Parameters """
        # Link lengths
        self.L12 = self.robot_cfg.links_length.L12  # from base to joint 2, [m] 0.2755
        self.L34 = self.robot_cfg.links_length.L34  # from joint 2 to joint 4, [m], 0.410
        self.L56 = self.robot_cfg.links_length.L56  # from joint 4 to joint 6, [m], 0.3111
        self.L78 = self.robot_cfg.links_length.L78  # from joint 6 to edge of EE where peg is attached, [m], 0.2188
        self.Ltip = self.robot_cfg.links_length.Ltip  # from edge of End-Effector to tip of peg, [m], 0.16

        """ Load Vortex Scene """
        # Define the setup and scene file paths
        self.setup_file = app_settings.vortex_resources_path / 'config_withgraphics.vxc'  # 'config_withoutgraphics.vxc'
        self.content_file = app_settings.vortex_resources_path / 'Kinova Gen2 Unjamming/kinova_gen2_sq_peg3dof.vxscene'

        # Create the Vortex Application
        self.vx_interface = VortexInterface()
        self.vx_interface.create_application(self.setup_file)

        self.vx_interface.load_scene(self.content_file)

        # Rendering
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # Create a display window
        self.vx_interface.load_display()
        self.vx_interface.render_display(active=(self.render_mode == 'human'))
        # self.vx_interface.render_display(active=False)

        # Initialize Robot position
        self.go_home()

        # Set parameters values. Done after going home so its not limited by joint torques
        self.vx_interface.set_app_mode(AppMode.EDITING)
        for field, field_value in {**self.joints_range, **self.forces_range}.items():
            self.vx_interface.set_parameter(field, field_value)

        # Save the state
        self.vx_interface.save_current_frame()
        self.vx_interface.set_app_mode(AppMode.SIMULATING)

        """ Finalize setup """
        self.sim_completed = False

        self.reset()

        print(f'Environment initialized. Time: {time.time() - init_start_time} sec')

    def _get_robot_config(self):
        """Load robot config from .yaml file"""
        config_path = app_settings.cfg_path / 'robot' / 'kinova_gen2.yaml'
        self.robot_cfg = OmegaConf.load(config_path)

    def _init_obs_space(self):
        """Observation space (12 observations: position, vel, ideal vel, torque, for each of 3 joints)"""
        self.obs = np.zeros(9)

        # Observation: [joint_positions, joint_velocities, joint_ideal_vel, joint_torques]
        # Each one is 1x(n_joints). Total size: 4*(n_joints)
        pos_min = [np.deg2rad(act.position_min) for act in self.robot_cfg.actuators.values()]
        pos_max = [np.deg2rad(act.position_max) for act in self.robot_cfg.actuators.values()]
        vel_min = [np.deg2rad(act.vel_min) for act in self.robot_cfg.actuators.values()]
        vel_max = [np.deg2rad(act.vel_max) for act in self.robot_cfg.actuators.values()]
        torque_min = [act.torque_min for act in self.robot_cfg.actuators.values()]
        torque_max = [act.torque_max for act in self.robot_cfg.actuators.values()]

        # Minimum and Maximum joint position limits (in deg)
        self.joints_range = {}
        self.joints_range['j2_pos_min'], self.joints_range['j2_pos_max'] = (pos_min[0], pos_max[0])
        self.joints_range['j4_pos_min'], self.joints_range['j4_pos_max'] = (pos_min[1], pos_max[1])
        self.joints_range['j6_pos_min'], self.joints_range['j6_pos_max'] = (pos_min[2], pos_max[2])

        # Minimum and Maximum joint force/torque limits (in N*m)
        self.forces_range = {}
        self.forces_range['j2_for_min'], self.forces_range['j2_for_max'] = (torque_min[0], torque_max[0])
        self.forces_range['j4_for_min'], self.forces_range['j4_for_max'] = (torque_min[1], torque_max[1])
        self.forces_range['j6_for_min'], self.forces_range['j6_for_max'] = (torque_min[2], torque_max[2])

        obs_low_bound = np.concatenate((pos_min, vel_min, vel_min, torque_min))
        obs_high_bound = np.concatenate((pos_max, vel_max, vel_max, torque_max))

        self.observation_space = spaces.Box(
            low=obs_low_bound,
            high=obs_high_bound,
            dtype=np.float32,
        )

    def _init_action_space(self):
        """Action space (2 actions: j2 aug, j6, aug)"""
        self.action = np.array([0.0, 0.0])  # Action outputed by RL
        self.command = np.array([0.0, 0.0, 0.0])  # Vel command to send to the robot
        self.next_j_vel = np.zeros(3)  # Next target vel
        self.prev_j_vel = np.zeros(3)  # Prev target vel

        act_low_bound = np.array([self.robot_cfg.actuators.j2.torque_min, self.robot_cfg.actuators.j6.torque_min])
        act_high_bound = np.array([self.robot_cfg.actuators.j2.torque_max, self.robot_cfg.actuators.j6.torque_max])
        self.action_space = spaces.Box(
            low=act_low_bound,
            high=act_high_bound,
            dtype=np.float32,
        )

    """ Actions """

    def go_home(self):
        """
        To bring the peg on top of the hole
        """
        self.vx_interface.set_app_mode(AppMode.SIMULATING)
        self.vx_interface.app.pause(False)

        """ Phase 1 """
        # Set joint velocities to initialize
        self.update_sim()

        j4_vel_id = (np.pi / 180.0) * 90.0 / self.t_init_step
        # j6_vel_id = self.insertion_misalign / self.t_init_step
        self.command = np.array([0.0, j4_vel_id, 0.0])
        # print(f'Start vel: {self._readJvel()}')

        # Step the Vortex simulation
        for i in range(self.init_steps):
            self.update_sim()

        # Read reference position and rotation
        th_current = self._readJpos()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X after Phase 1:')
        print(pos_current[0])

        # Phase 1 pause
        self.command = np.array([0.0, 0.0, 0.0])

        for i in range(self.pause_steps):
            self.update_sim()

        """ Phase 2 (move downwards quickly and also make the tips aligned with the hole) """
        for i in range(self.pre_insert_steps):
            th_current = self._readJpos()
            self.cur_j_vel = self._get_ik_vels(self.pre_insertz, i, step_types=self.pre_insert_steps)
            self.command = self.cur_j_vel

            self.update_sim()

        th_current = self._readJpos()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X after Phase 2:')
        print(pos_current[0])

        # Phase 2 pause
        self.command = np.array([0.0, 0.0, 0.0])

        for i in range(self.pause_steps):
            self.update_sim()

    def step(self, action):
        """
        Take one step. This is the main function of the environment.

        The state of the robot is defined as all the measurements that the physical robot could obtain from sensors:
        - position
        - vel
        - ideal vel
        - torque

        The info returned is the other information that might be useful for analysis, but not for learning:
        - command
        - plug force
        - plug torque


        Args:
            action (np.Array): The action to take. Defined as a correction to the joint velocities

        Returns:
            obs (np.Array): The observation of the environment after taking the step
            reward (float): The reward obtained after taking the step
            sim_completed (bool): Flag indicating if the simulation is completed
            done (bool): Flag indicating if the episode is done
            info (dict): Additional information about the step
        """
        self.action = action
        self.prev_j_vel = self.next_j_vel
        self.next_j_vel = self._get_ik_vels(self.insertz, self.step_count, step_types=self.insertion_steps)

        j2_vel = self.next_j_vel[0] - self.action_coeff * action[0]
        j4_vel = self.next_j_vel[1] + self.action_coeff * action[1] - self.action_coeff * action[0]
        j6_vel = self.next_j_vel[2] + self.action_coeff * action[1]

        # Apply actions
        self.command = np.array([j2_vel, j4_vel, j6_vel])

        # Step the simulation
        self.update_sim()

        # Observations
        self.obs = self._get_robot_state()

        # Reward
        reward = self._get_reward()

        # Done flag
        self.step_count += 1
        if self.step_count >= self.insertion_steps:
            self.sim_completed = True

        # Info
        info = self._get_step_info()  # plug force and torque

        return self.obs, reward, self.sim_completed, False, info

    def render(self):
        if self.render_mode is None:
            self.vx_interface.render_display(False)
        elif self.render_mode == 'human':
            self.vx_interface.render_display(True)

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Returns the robot states after the reset and information about the reset.

        env_info:
        - TODO: Add info about the reset
            - Insertion misalignment
            - Friction coefficient
            ...

        Args:
            seed (_type_, optional): Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            obs: Robot state
            env_info (dict): Information about the env after reset
        """
        logger.debug('Reseting env')

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Random parameters
        self.insertion_misalign = (np.pi / 180.0) * (
            np.random.uniform(self.min_misalign, self.max_misalign)
        )  # joint 6 misalignment, negative and positive
        # print('Insertion misalignment: ' + str(self.insertion_misalign))

        # Reset Robot
        self.vx_interface.reset_saved_frame()

        # Reset parameters
        self.step_count = 0  # Step num in the episode
        self.sim_completed = False

        env_info = {
            'insertion_misalignment': self.insertion_misalign,
        }

        return self._get_robot_state(), env_info

    def update_sim(self):
        """To update the state of the robot.

        Sends the action and reads the robot's state

        """

        self._send_joint_target_vel(self.command)

        self.vx_interface.app.update()

        self.sim_time = self.vx_interface.app.getSimulationTime()
        self.obs = self._get_robot_state()

    """ Vortex interface functions """

    def _get_robot_state(self) -> np.array:
        """Observation space - 12 observations:
            - position
            - vel
            - ideal vel
            - torque

        , for each of 3 joints

        Returns:
            np.array: Robot state
        """
        joint_poses = self._readJpos()
        joint_vel = self._readJvel()
        joint_torques = self._readJtorque()
        joint_ideal_vel = self.next_j_vel

        return np.concatenate((joint_poses, joint_vel, joint_ideal_vel, joint_torques), dtype=np.float32)

    def _get_step_info(self) -> dict:
        """Get info about the robot
        - Command
        - Plug force
        - Plug torque
        - Insertion depth

        Returns:
            dict: Info about the robot
        """
        info = {
            'command': self.command,
            'plug_force': self._get_plug_force(),
            'plug_torque': self._get_plug_torque(),
            'insertion_depth': self._get_insertion_depth(),
        }

        return info

    def _readJpos(self):
        j2_pos = self.vx_interface.get_output(VX_OUT.j2_pos_real)
        j4_pos = self.vx_interface.get_output(VX_OUT.j4_pos_real)
        j6_pos = self.vx_interface.get_output(VX_OUT.j6_pos_real)

        return np.array([j2_pos, j4_pos, j6_pos], dtype=np.float32)

    def _readJvel(self):
        j2_vel = self.vx_interface.get_output(VX_OUT.j2_vel_real)
        j4_vel = self.vx_interface.get_output(VX_OUT.j4_vel_real)
        j6_vel = self.vx_interface.get_output(VX_OUT.j6_vel_real)

        return np.array([j2_vel, j4_vel, j6_vel], dtype=np.float32)

    def _readJtorque(self):
        j2_t = self.vx_interface.get_output(VX_OUT.j2_torque)
        j4_t = self.vx_interface.get_output(VX_OUT.j4_torque)
        j6_t = self.vx_interface.get_output(VX_OUT.j6_torque)

        return np.array([j2_t, j4_t, j6_t], dtype=np.float32)

    def _readJvel_target(self):
        j2_target = self.vx_interface.get_input(VX_IN.j2_vel_id)
        j4_target = self.vx_interface.get_input(VX_IN.j4_vel_id)
        j6_target = self.vx_interface.get_input(VX_IN.j6_vel_id)

        return np.array([j2_target, j4_target, j6_target])

    def _get_plug_force(self) -> np.array:
        """Read plug force

        Returns:
            np.array(1x3): [x, y, z]
        """
        plug_force = self.vx_interface.get_output(VX_OUT.plug_force)
        return np.array([plug_force.x, plug_force.y, plug_force.z])

    def _get_plug_torque(self) -> np.array:
        """Read plug torque

        Returns:
            np.array: [x, y, z]
        """
        plug_torque = self.vx_interface.get_output(VX_OUT.plug_torque)
        return np.array([plug_torque.x, plug_torque.y, plug_torque.z])

    def _send_joint_target_vel(self, target_vels):
        self.vx_interface.set_input(VX_IN.j2_vel_id, target_vels[0])
        self.vx_interface.set_input(VX_IN.j4_vel_id, target_vels[1])
        self.vx_interface.set_input(VX_IN.j6_vel_id, target_vels[2])

    """ Utilities """

    def _get_reward(self):
        j2_id = self.next_j_vel[0]
        j4_id = self.next_j_vel[1]
        j6_id = self.next_j_vel[2]

        #  reward = self.reward_weight*(-abs((shv_id-shv)*self.shoulder_torque.value)-abs((elv_id-elv)*self.elbow_torque.value)-abs((wrv_id-wrv)*self.wrist_torque.value))
        joint_vels = self.obs[3:6]
        joint_torques = self.obs[0:3]

        reward = self.reward_weight * (
            -abs((j2_id - joint_vels[0]) * joint_torques[0])
            - abs((j4_id - joint_vels[1]) * joint_torques[1])
            - abs((j6_id - joint_vels[2]) * joint_torques[2])
        )
        return reward

    def _get_ik_vels(self, down_speed, cur_count, step_types):
        th_current = self._readJpos()
        current_pos = self._read_tips_pos_fk(th_current)
        if step_types == self.pre_insert_steps:
            x_set = current_pos[0] - self.xpos_hole
            x_vel = -x_set / (self.h)
            z_vel = down_speed / (step_types * self.h)

        elif step_types == self.insertion_steps:
            vel = down_speed / (step_types * self.h)
            x_vel = vel * np.sin(self.insertion_misalign)
            z_vel = vel * np.cos(self.insertion_misalign)

        else:
            print('STEP TYPES DOES NOT MATCH')

        rot_vel = 0.0
        next_vel = [x_vel, -z_vel, rot_vel]
        J = self._build_Jacobian(th_current)
        Jinv = np.linalg.inv(J)
        j_vel_next = np.dot(Jinv, next_vel)
        return j_vel_next

    def _read_tips_pos_fk(self, th_current):
        q2 = th_current[0]
        q4 = th_current[1]
        q6 = th_current[2]

        current_tips_posx = (
            self.L34 * np.sin(-q2)
            + self.L56 * np.sin(-q2 + q4)
            + self.L78 * np.sin(-q2 + q4 - q6)
            + self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        current_tips_posz = (
            self.L12
            + self.L34 * np.cos(-q2)
            + self.L56 * np.cos(-q2 + q4)
            + self.L78 * np.cos(-q2 + q4 - q6)
            + self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        current_tips_rot = -q2 + q4 - q6 + 90.0 * (np.pi / 180.0)

        return current_tips_posx, current_tips_posz, current_tips_rot

    def _build_Jacobian(self, th_current):
        q2 = th_current[0]
        q4 = th_current[1]
        q6 = th_current[2]

        a_x = (
            -self.L34 * np.cos(-q2)
            - self.L56 * np.cos(-q2 + q4)
            - self.L78 * np.cos(-q2 + q4 - q6)
            - self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        b_x = (
            self.L56 * np.cos(-q2 + q4)
            + self.L78 * np.cos(-q2 + q4 - q6)
            + self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)
        )
        c_x = -self.L78 * np.cos(-q2 + q4 - q6) - self.Ltip * np.cos(-q2 + q4 - q6 + np.pi / 2.0)

        a_z = (
            self.L34 * np.sin(-q2)
            + self.L56 * np.sin(-q2 + q4)
            + self.L78 * np.sin(-q2 + q4 - q6)
            + self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        b_z = (
            -self.L56 * np.sin(-q2 + q4)
            - self.L78 * np.sin(-q2 + q4 - q6)
            - self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)
        )
        c_z = self.L78 * np.sin(-q2 + q4 - q6) + self.Ltip * np.sin(-q2 + q4 - q6 + np.pi / 2.0)

        J = [[a_x, b_x, c_x], [a_z, b_z, c_z], [-1.0, 1.0, -1.0]]

        return J

    def _get_insertion_depth(self):
        th_current = self._readJpos()
        x, z, rot = self._read_tips_pos_fk(th_current)
        return np.array([x, z, rot])
