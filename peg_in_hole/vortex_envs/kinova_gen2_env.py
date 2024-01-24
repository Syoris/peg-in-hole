import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ctypes
from pathlib import Path
from pydantic import BaseModel
import logging
from omegaconf import OmegaConf, DictConfig

from peg_in_hole.settings import app_settings
from peg_in_hole.vortex_envs.vortex_interface import VortexInterface, AppMode
from peg_in_hole.vortex_envs.robot_schema_config import KinovaConfig
from peg_in_hole.utils.Neptune import NeptuneRun

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


class KinovaGen2Env(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, neptune_logger=None, env_cfg=None):
        """Load config"""
        self._get_robot_config()

        # General params
        self.neptune_logger = neptune_logger
        self.sim_time = 0.0
        self.episode = 0
        self.episode_logger = self.neptune_logger.run[f'episode/{self.episode}']
        self.ep_history = {
            'step': [],
            'sim_time': [],
            'obs': [],
            'command': [],
            'action': [],
        }  # Save here the states at each log_freq

        """ Observation space (12 observations: position, vel, ideal vel, torque, for each of 3 joints) """
        self.obs = np.zeros(12)

        # Observation: [joint_positions, joint_velocities, joint_ideal_vel, joint_torques]
        # Each one is 1x(n_joints). Total size: 4*(n_joints)
        pos_min = [act.position_min for act in self.robot_cfg.actuators.values()]
        pos_max = [act.position_max for act in self.robot_cfg.actuators.values()]
        vel_min = [act.vel_min for act in self.robot_cfg.actuators.values()]
        vel_max = [act.vel_max for act in self.robot_cfg.actuators.values()]
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
            dtype=np.float64,
        )

        """ Action space (2 actions: j2 aug, j6, aug) """
        self.action = np.array([0.0, 0.0])  # Action outputed by RL
        self.command = np.array([0.0, 0.0, 0.0])  # Vel command to send to the robot

        self.next_j_vel = np.zeros(3)  # Next target vel
        self.prev_j_vel = np.zeros(3)  # Prev target vel

        act_low_bound = np.array([self.robot_cfg.actuators.j2.torque_min, self.robot_cfg.actuators.j6.torque_min])
        act_high_bound = np.array([self.robot_cfg.actuators.j2.torque_max, self.robot_cfg.actuators.j6.torque_max])
        self.action_space = spaces.Box(
            low=act_low_bound,
            high=act_high_bound,
            dtype=np.float64,
        )

        """ RL Hyperparameters """
        self.action_coeff = 0.01  # coefficient the action will be multiplied by

        # Reward
        self.reward_min_threshold = -5.0  # NOT CURRENTLY USED
        self.min_height_threshold = 0.005  # NOT CURRENTLY USED
        self.reward_weight = 0.04

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
        self.nStep = 0  # Step counter
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
        self.nStep = 0

        self.reset()

        self.episode += 1
        self.episode_logger = self.neptune_logger.run[f'episode/{self.episode}']

    def _get_robot_config(self):
        config_path = app_settings.cfg_path / 'robot' / 'kinova_gen2.yaml'
        # config_path = 'cfg/tasks/Insert_Kinova3DoF.yaml'
        self.robot_cfg = OmegaConf.load(config_path)

    def go_home(self):
        """
        To bring the peg on top of the hole
        """
        self.vx_interface.set_app_mode(AppMode.SIMULATING)
        self.vx_interface.app.pause(False)

        """ Phase 1 """
        # Set joint velocities to initialize
        self.update()

        j4_vel_id = (np.pi / 180.0) * 90.0 / self.t_init_step
        # j6_vel_id = self.insertion_misalign / self.t_init_step
        self.command = np.array([0.0, j4_vel_id, 0.0])
        # print(f'Start vel: {self._readJvel()}')

        # Step the Vortex simulation
        for i in range(self.init_steps):
            self.update()

        # Read reference position and rotation
        th_current = self._readJpos()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X after Phase 1:')
        print(pos_current[0])

        # Phase 1 pause
        self.command = np.array([0.0, 0.0, 0.0])

        for i in range(self.pause_steps):
            self.update()

        """ Phase 2 (move downwards quickly and also make the tips aligned with the hole) """
        for i in range(self.pre_insert_steps):
            th_current = self._readJpos()
            self.cur_j_vel = self._get_ik_vels(self.pre_insertz, i, step_types=self.pre_insert_steps)
            self.command = self.cur_j_vel

            self.update()

        th_current = self._readJpos()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X after Phase 2:')
        print(pos_current[0])

        # Phase 2 pause
        self.command = np.array([0.0, 0.0, 0.0])

        for i in range(self.pause_steps):
            self.update()

    def step(self, action):
        self.action = action
        self.prev_j_vel = self.next_j_vel
        self.next_j_vel = self._get_ik_vels(self.insertz, self.nStep, step_types=self.insertion_steps)

        j2_vel = self.next_j_vel[0] - self.action_coeff * action[0]
        j4_vel = self.next_j_vel[1] + self.action_coeff * action[1] - self.action_coeff * action[0]
        j6_vel = self.next_j_vel[2] + self.action_coeff * action[1]

        # Apply actions
        self.command = np.array([j2_vel, j4_vel, j6_vel])

        # Step the simulation
        self.update()

        # Observations
        self.obs = self._get_obs()

        # Reward
        reward = self._get_reward()

        # Done flag
        self.nStep += 1
        if self.nStep >= self.insertion_steps:
            self.sim_completed = True

        return self.obs, reward, self.sim_completed, False, {}

    def render(self):
        self.vx_interface.render_display()

    def reset(self, seed=None, options=None):
        logger.debug('Reseting env')

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Log results to neptune
        # self._log_ep_data()

        # Random parameters
        self.insertion_misalign = (np.pi / 180.0) * (
            np.random.uniform(self.min_misalign, self.max_misalign)
        )  # joint 6 misalignment, negative and positive
        print('Insertion misalignment: ' + str(self.insertion_misalign))

        # Reset Robot
        self.vx_interface.reset_saved_frame()

        # Reset parameters
        self.nStep = 0
        self.sim_completed = False
        self.episode += 1

        info = {}

        return self._get_obs(), info

    """ Vortex interface functions """

    def update(self):
        """To update the state of the robot.

        Sends the action and reads the robot's state

        """

        self._send_joint_target_vel(self.command)

        self.vx_interface.app.update()

        self.sim_time = self.vx_interface.app.getSimulationTime()
        self.obs = self._get_obs()

        log_dict = {
            'sim_time': self.sim_time,
            'step': self.nStep,
            'obs': self.obs,
            'command': self.command,
            'action': self.action,
        }

        for param, val in log_dict.items():
            self.ep_history[param].append(val)

    def _get_obs(self) -> np.array:
        """Observation space - 12 observations:
            - position
            - vel
            - ideal vel
            - torque

        , for each of 3 joints

        Returns:
            np.array: Observation
        """
        joint_poses = self._readJpos()
        joint_vel = self._readJvel()
        joint_torques = self._readJtorque()
        joint_ideal_vel = self.next_j_vel

        return np.concatenate((joint_poses, joint_vel, joint_ideal_vel, joint_torques))

    def _readJpos(self):
        j2_pos = self.vx_interface.get_output(VX_OUT.j2_pos_real)
        j4_pos = self.vx_interface.get_output(VX_OUT.j4_pos_real)
        j6_pos = self.vx_interface.get_output(VX_OUT.j6_pos_real)

        return np.array([j2_pos, j4_pos, j6_pos])

    def _readJvel(self):
        j2_vel = self.vx_interface.get_output(VX_OUT.j2_vel_real)
        j4_vel = self.vx_interface.get_output(VX_OUT.j4_vel_real)
        j6_vel = self.vx_interface.get_output(VX_OUT.j6_vel_real)

        return np.array([j2_vel, j4_vel, j6_vel])

    def _readJtorque(self):
        j2_t = self.vx_interface.get_output(VX_OUT.j2_torque)
        j4_t = self.vx_interface.get_output(VX_OUT.j4_torque)
        j6_t = self.vx_interface.get_output(VX_OUT.j6_torque)

        return np.array([j2_t, j4_t, j6_t])

    def _readJvel_target(self):
        j2_target = self.vx_interface.get_input(VX_IN.j2_vel_id)
        j4_target = self.vx_interface.get_input(VX_IN.j4_vel_id)
        j6_target = self.vx_interface.get_input(VX_IN.j6_vel_id)

        return np.array([j2_target, j4_target, j6_target])

    def get_plug_force(self):
        plug_force = self.vx_interface.get_output(VX_OUT.plug_force)
        return plug_force

    def get_plug_torque(self):
        plug_torque = self.vx_interface.get_output(VX_OUT.plug_torque)
        return plug_torque

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

    def get_insertion_depth(self):
        th_current = self._readJpos()
        x, z, rot = self._read_tips_pos_fk(th_current)
        return x, z, rot

    def _log_ep_data(self):
        ep_logger = self.neptune_logger.run[f'episode/{self.episode}']

        obs = np.vstack(self.ep_history['obs'])
        command = np.vstack(self.ep_history['command'])
        action = np.vstack(self.ep_history['action'])

        log_dict = {
            'sim_time': self.ep_history['sim_time'],
            'step': self.ep_history['step'],
            'j2_pos': obs[:, 0],
            'j4_pos': obs[:, 1],
            'j6_pos': obs[:, 2],
            'j2_vel': obs[:, 0 + 3],
            'j4_vel': obs[:, 1 + 3],
            'j6_vel': obs[:, 2 + 3],
            'j2_ideal_vel': obs[:, 0 + 6],
            'j4_ideal_vel': obs[:, 1 + 6],
            'j6_ideal_vel': obs[:, 2 + 6],
            'j2_torque': obs[:, 0 + 9],
            'j4_torque': obs[:, 1 + 9],
            'j6_torque': obs[:, 2 + 9],
            'j2_cmd': command[:, 0],
            'j4_cmd': command[:, 1],
            'j6_cmd': command[:, 2],
            'j2_act': action[:, 0],
            'j6_act': action[:, 1],
        }

        for param, val in log_dict.items():
            ep_logger[param].extend(list(val))
