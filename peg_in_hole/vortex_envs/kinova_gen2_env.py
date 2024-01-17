import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ctypes
from settings import app_settings
from pathlib import Path
from pydantic import BaseModel
import logging

from peg_in_hole.vortex_envs.vortex_interface import VortexInterface, AppMode

logger = logging.getLogger(__name__)

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


VX_IN = VX_Inputs()


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


VX_OUT = VX_Outputs()

""" Kinova Robot Interface """


class KinovaGen2Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Observation space (9 observations: torque, ideal velocity, actual velocity, for each of 3 joints)
        """
        # TODO: To yaml and pydantic dataclass
        self.max_j2_t, self.min_j2_t = 500.0, -500.0
        self.max_j4_t, self.min_j4_t = 500.0, -500.0
        self.max_j6_t, self.min_j6_t = 500.0, -500.0
        self.max_j2_v, self.min_j2_v = 1.0, -1.0
        self.max_j4_v, self.min_j4_v = 1.0, -1.0
        self.max_j6_v, self.min_j6_v = 1.0, -1.0
        self.max_j2_v_id, self.min_j2_v_id = 1.0, -1.0
        self.max_j4_v_id, self.min_j4_v_id = 1.0, -1.0
        self.max_j6_v_id, self.min_j6_v_id = 1.0, -1.0
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    self.min_j2_t,
                    self.min_j4_t,
                    self.min_j6_t,
                    self.min_j2_v,
                    self.min_j4_v,
                    self.min_j6_v,
                    self.min_j2_v_id,
                    self.min_j4_v_id,
                    self.min_j6_v_id,
                ]
            ),
            high=np.array(
                [
                    self.max_j2_t,
                    self.max_j4_t,
                    self.max_j6_t,
                    self.max_j2_v,
                    self.max_j4_v,
                    self.max_j6_v,
                    self.max_j2_v_id,
                    self.max_j4_v_id,
                    self.max_j6_v_id,
                ]
            ),
            shape=(9,),
            dtype=np.float32,
        )

        # Action Space
        ...

        # Reward
        ...

        self.next_j_vel = np.zeros(3)
        self.prev_j_vel = np.zeros(3)

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

        # TODO: To YAML and pydantic data class
        # Minimum and Maximum joint position limits (in radians)
        self.joints_range = {}
        self.joints_range['j2_pos_min'], self.joints_range['j2_pos_max'] = (
            -360,  # * (np.pi / 180.0),
            360,  # * (np.pi / 180.0),
        )
        self.joints_range['j4_pos_min'], self.joints_range['j4_pos_max'] = (
            -360,  # * (np.pi / 180.0),
            360,  # * (np.pi / 180.0),
        )
        self.joints_range['j6_pos_min'], self.joints_range['j6_pos_max'] = (
            -360,  # * (np.pi / 180.0),
            360,  # * (np.pi / 180.0),
        )

        # Minimum and Maximum joint force/torque limits (in N*m)
        self.forces_range = {}
        self.forces_range['j2_for_min'], self.forces_range['j2_for_max'] = -14.0, 14.0
        self.forces_range['j4_for_min'], self.forces_range['j4_for_max'] = -5.0, 5.0
        self.forces_range['j6_for_min'], self.forces_range['j6_for_max'] = -2.0, 2.0

        # Link lengths #
        self.L12 = 0.2755  # from base to joint 2, in metres
        self.L34 = 0.410  # from joint 2 to joint 4, in metres
        self.L56 = 0.3111  # from joint 4 to joint 6, in metres
        self.L78 = 0.2188  # from joint 6 to edge of End-Effector where peg is attached, in metres
        self.Ltip = 0.16  # from edge of End-Effector to tip of peg, in metres

        # # Define the setup and mechanism file paths
        self.setup_file = app_settings.vortex_resources_path / 'config_withgraphics.vxc'  # 'config_withoutgraphics.vxc'
        # self.setup_file = env_folder_rel_path / 'config_withoutgraphics.vxc'  # 'config_withoutgraphics.vxc'

        self.content_file = app_settings.vortex_resources_path / 'Kinova Gen2 Unjamming/kinova_gen2_sq_peg3dof.vxscene'

        # # Create the Vortex Application
        self.vx_interface = VortexInterface()
        self.vx_interface.create_application(self.setup_file)

        self.vx_interface.load_scene(self.content_file)

        # Create a display window
        self.vx_interface.load_display()

        # Initialize Robot position
        self.go_to_home()

        # Set parameters values
        self.vx_interface.set_app_mode(AppMode.EDITING)
        for field, field_value in {**self.joints_range, **self.forces_range}.items():
            self.vx_interface.set_parameter(field, field_value)

        self.vx_interface.save_current_frame()

        self.vx_interface.set_app_mode(AppMode.SIMULATING)

    def go_to_home(self):
        """
        To bring the peg on top of the hole
        """
        self.vx_interface.set_app_mode(AppMode.SIMULATING)
        self.vx_interface.app.pause(False)

        """Phase 1"""
        """ set joint velocities to initialize """
        print(self._readJvel_target())

        j4_vel_id = (np.pi / 180.0) * 90.0 / self.t_init_step
        # j6_vel_id = self.insertion_misalign / self.t_init_step
        # self.vx_interface.set_input(VX_IN.j4_vel_id, j4_vel_id)
        self._send_joint_target_vel(np.array([0.0, j4_vel_id, 0.0]))

        # print(f'Vel target: {self._readJvel_target()}')
        # print(f'Start torque: {self._readJtorque()}')
        # print(f'Start vel: {self._readJvel()}')

        """ step the Vortex simulation """
        for i in range(self.init_steps):
            self.vx_interface.app.update()

            obs = self._readJtorque()
            width = 10
            precision = 4
            print(f'{obs[0]:^{width}.{precision}f} | {obs[1]:^{width}.{precision}f} | {obs[2]:^{width}.{precision}f}')

        """ Read reference position and rotation """
        th_current = self._readJpos()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X after Phase 1:')
        print(pos_current[0])

        """ Phase 1 pause """
        self.vx_interface.set_input(VX_IN.j4_vel_id, 0)
        self.vx_interface.set_input(VX_IN.j6_vel_id, 0)

        for i in range(self.pause_steps):
            self.vx_interface.app.update()

        """ Phase 2 (move downwards quickly and also make the tips aligned with the hole) """
        for i in range(self.pre_insert_steps):
            th_current = self._readJpos()
            self.cur_j_vel = self._get_ik_vels(self.pre_insertz, i, step_types=self.pre_insert_steps)

            self._send_joint_target_vel(self.cur_j_vel)

            self.vx_interface.app.update()

            obs = self._readJtorque()
            width = 10
            precision = 4
            print(f'{obs[0]:^{width}.{precision}f} | {obs[1]:^{width}.{precision}f} | {obs[2]:^{width}.{precision}f}')

        th_current = self._readJpos()
        pos_current = self._read_tips_pos_fk(th_current)
        print('X after Phase 2:')
        print(pos_current[0])

        """ Phase 2 pause  """
        self._send_joint_target_vel(np.array([0.0, 0.0, 0.0]))

        for i in range(self.pause_steps):
            self.vx_interface.app.update()

    def step(self, action):
        # Apply actions
        self.vx_interface.set_input('j2_vel_id', action[0])
        self.vx_interface.set_input('j4_vel_id', action[1])
        self.vx_interface.set_input('j6_vel_id', action[2])

        # Step the simulation
        self.vx_interface.app.update()

        # Observations
        obs = self._get_obs()

        # Done flag
        ...

        # Reward
        reward = ...

        return obs

    def render(self):
        self.vx_interface.render_display()

    def reset(self):
        self.vx_interface.reset_saved_frame()

    """ Vortex interface functions """

    def _get_obs(self):
        # self.joint_torques = self._readJtorque()
        # self.joint_vel_real = self._readJvel()
        # joint_poses = self._readJpos()
        # joint_vel_id = []
        # joint_vel_id.append(self.next_j_vel[0])
        # joint_vel_id.append(self.next_j_vel[1])
        # joint_vel_id.append(self.next_j_vel[2])

        # print('Observation')
        # print(f'joint_poses: {joint_poses}')
        # print(f'\njoint_torques: {self.joint_torques}')
        # print(f'\nJoint vels: {self.joint_vel_real}')
        # print(f'\njoint_vel_id: {joint_vel_id}')

        # FIND OUT IF TORQUES ARE SURPASSING LIMITS #
        # if abs(self.joint_torques[0])>14.0:
        #   print("Joint 2 torque: {}".format(self.joint_torques[0]))
        # if abs(self.joint_torques[1])>5.0:
        #   print("Joint 4 torque: {}".format(self.joint_torques[1]))
        # if abs(self.joint_torques[2])>2.0:
        #   print("Joint 6 torque: {}".format(self.joint_torques[2]))

        j2_pos_real = self.vx_interface.get_output('j2_pos_real')
        j4_pos_real = self.vx_interface.get_output('j2_pos_real')
        j6_pos_real = self.vx_interface.get_output('j2_pos_real')

        # return np.concatenate((self.joint_torques, self.joint_vel_real, joint_vel_id))
        return [j2_pos_real, j4_pos_real, j6_pos_real]

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

    def _get_plug_force(self):
        plug_force = self.vx_interface.get_output(VX_OUT.plug_force)
        return plug_force

    def _get_plug_torque(self):
        plug_torque = self.vx_interface.get_output(VX_OUT.plug_torque)
        return plug_torque

    def _send_joint_target_vel(self, target_vels):
        self.vx_interface.set_input(VX_IN.j2_vel_id, target_vels[0])
        self.vx_interface.set_input(VX_IN.j4_vel_id, target_vels[1])
        self.vx_interface.set_input(VX_IN.j6_vel_id, target_vels[2])

    """ Utilities """

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