import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ctypes
from settings import APP_SETTINGS
from pathlib import Path


class Vector3(ctypes.Structure):
    _fields_ = ('x', ctypes.c_double), ('y', ctypes.c_double), ('z', ctypes.c_double)

    def __repr__(self):
        return '({0}, {1}, {2})'.format(self.x, self.y, self.z)


class Vector4(ctypes.Structure):
    _fields_ = ('x', ctypes.c_double), ('y', ctypes.c_double), ('z', ctypes.c_double), ('w', ctypes.c_double)


class KinovaGen2Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Observation space (9 observations: torque, ideal velocity, actual velocity, for each of 3 joints)
        """
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

        # Minimum and Maximum joint position limits (in radians)
        self.j2_pos_min, self.j2_pos_max = (
            47.0 * (np.pi / 180.0),
            313.0 * (np.pi / 180.0),
        )
        self.j4_pos_min, self.j4_pos_max = (
            30.0 * (np.pi / 180.0),
            330.0 * (np.pi / 180.0),
        )
        self.j6_pos_min, self.j6_pos_max = (
            65.0 * (np.pi / 180.0),
            295.0 * (np.pi / 180.0),
        )

        # Minimum and Maximum joint force/torque limits (in N*m)
        self.j2_for_min, self.j2_for_max = -14.0, 14.0
        self.j4_for_min, self.j4_for_max = -5.0, 5.0
        self.j6_for_min, self.j6_for_max = -2.0, 2.0

        # Link lengths #
        self.L12 = 0.2755  # from base to joint 2, in metres
        self.L34 = 0.410  # from joint 2 to joint 4, in metres
        self.L56 = 0.3111  # from joint 4 to joint 6, in metres
        self.L78 = 0.2188  # from joint 6 to edge of End-Effector where peg is attached, in metres
        self.Ltip = 0.16  # from edge of End-Effector to tip of peg, in metres

        # # VxMechanism variable for the mechanism to be loaded.
        # self.vxmechanism = None
        # self.mechanism = None
        # self.interface = None

        # # Define the setup and mechanism file paths
        env_folder_rel_path = Path('src/vortex_envs/vortex_resources')
        self.setup_file = env_folder_rel_path / 'config_withoutgraphics.vxc'
        self.content_file = env_folder_rel_path / 'Kinova Gen2 Unjamming/kinova_gen2_sq_peg3dof.vxscene'

        # # Create the Vortex Application
        # self.application = vxatp3.VxATPConfig.createApplication(self, 'Peg-in-hole App', self.setup_file)

        # # Create a display window
        # self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
        # self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue('Windowed')
        # self.display.setName('3D Display')
        # self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(50, 50, 1280, 720))

        ##
        dll_path = APP_SETTINGS.vortex_installation_path / 'bin' / 'VortexIntegration.dll'
        vxDLL = ctypes.WinDLL(str(dll_path))

        """ declare function inputs and outputs so that they are called correctly (necessary for Python 3 calling C functions) """
        vxDLL.VortexLoadScene.restype = ctypes.c_void_p
        vxDLL.VortexGetChildByName.restype = ctypes.c_void_p
        vxDLL.VortexGetChildByName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        vxDLL.VortexSetInputReal.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
        vxDLL.VortexGetOutputReal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        vxDLL.VortexGetOutputVector3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(Vector3),
        ]
        vxDLL.VortexGetOutputMatrix.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(Vector3),
            ctypes.POINTER(Vector4),
        ]
        vxDLL.VortexUnloadScene.argtypes = [ctypes.c_void_p]

        self.vxDLL = vxDLL
        # setup_path = 'src/vortex_envs/vortex_resources/config_withoutgraphics.vxc'
        setup_path = (
            'C:/Users/charl/Local Documents/git/peg-in-hole/src/vortex_envs/vortex_resources/config_withoutgraphics.vxc'
        )

        setup_path = '/src/vortex_envs/vortex_resources/config_withoutgraphics.vxc'

        # self.vxDLL.VortexCreateApplication(setup_path.encode('ascii'), '', '', '', None)

        self.vxDLL.VortexCreateApplication(str(self.setup_file).encode('ascii'), '', '', '', None)

        self.scene = self.vxDLL.VortexLoadScene(str(self.content_file).encode('ascii'))

        if self.scene is None or self.scene == 0:
            raise RuntimeError('Scene not properly loaded')

        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_pos_min'.encode('ascii'),
            self.j2_pos_min,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_pos_max'.encode('ascii'),
            self.j2_pos_max,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_pos_min'.encode('ascii'),
            self.j4_pos_min,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_pos_max'.encode('ascii'),
            self.j4_pos_max,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_pos_min'.encode('ascii'),
            self.j6_pos_min,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_pos_max'.encode('ascii'),
            self.j6_pos_max,
        )

        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_for_min'.encode('ascii'),
            self.j2_for_min,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_for_max'.encode('ascii'),
            self.j2_for_max,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_for_min'.encode('ascii'),
            self.j4_for_min,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_for_max'.encode('ascii'),
            self.j4_for_max,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_for_min'.encode('ascii'),
            self.j6_for_min,
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_for_max'.encode('ascii'),
            self.j6_for_max,
        )

        if self.scene != 0:
            """ switch to simulation mode """
            self.vxDLL.VortexSetApplicationMode(
                0, True
            )  # 0: switch to simulation mode, True: wait until application has changed modes
            self.vxDLL.VortexPause(False)

            """ Phase 1 """
            """ set joint velocities to initialize """
            j4_vel_id = (np.pi / 180.0) * 90.0 / self.t_init_step
            # j6_vel_id = self.insertion_misalign / self.t_init_step
            self.vxDLL.VortexSetInputReal(
                self.scene,
                'ML Interface'.encode('ascii'),
                'j4_vel_id'.encode('ascii'),
                j4_vel_id,
            )
            # self.vxDLL.VortexSetInputReal(self.scene, "ML Interface".encode('ascii'), "j6_vel_id".encode('ascii'), j6_vel_id)
            """ step the Vortex simulation """
            for i in range(self.init_steps):
                self.vxDLL.VortexUpdateApplication()
            """ read reference position and rotation """
            # th_current = self._readJpos()
            # pos_current = self._read_tips_pos_fk(th_current)
            # print("X after Phase 1:")
            # print(pos_current[0])

            """ Phase 1 pause """
            self.vxDLL.VortexSetInputReal(
                self.scene,
                'ML Interface'.encode('ascii'),
                'j4_vel_id'.encode('ascii'),
                0.0,
            )
            self.vxDLL.VortexSetInputReal(
                self.scene,
                'ML Interface'.encode('ascii'),
                'j6_vel_id'.encode('ascii'),
                0.0,
            )
            for i in range(self.pause_steps):
                self.vxDLL.VortexUpdateApplication()
            """ read reference position and rotation """
            # th_current = self._readJpos()
            # self.ref_pos, self.ref_rot = self._read_ee_pos_rot_fk(th_current)

            """ Phase 2 (move downwards quickly and also make the tips aligned with the hole) """
            for i in range(self.pre_insert_steps):
                th_current = self._readJpos()
                self.cur_j_vel = self._get_ik_vels(self.pre_insertz, i, step_types=self.pre_insert_steps)
                self.vxDLL.VortexSetInputReal(
                    self.scene,
                    'ML Interface'.encode('ascii'),
                    'j2_vel_id'.encode('ascii'),
                    self.cur_j_vel[0],
                )
                self.vxDLL.VortexSetInputReal(
                    self.scene,
                    'ML Interface'.encode('ascii'),
                    'j4_vel_id'.encode('ascii'),
                    self.cur_j_vel[1],
                )
                self.vxDLL.VortexSetInputReal(
                    self.scene,
                    'ML Interface'.encode('ascii'),
                    'j6_vel_id'.encode('ascii'),
                    self.cur_j_vel[2],
                )
                self.vxDLL.VortexUpdateApplication()
                self._get_obs()

            # th_current = self._readJpos()
            # pos_current = self._read_tips_pos_fk(th_current)
            # print("X after Phase 1:")
            # print(pos_current[0])

            """ Phase 2 pause  """
            self.vxDLL.VortexSetInputReal(
                self.scene,
                'ML Interface'.encode('ascii'),
                'j2_vel_id'.encode('ascii'),
                0.0,
            )
            self.vxDLL.VortexSetInputReal(
                self.scene,
                'ML Interface'.encode('ascii'),
                'j4_vel_id'.encode('ascii'),
                0.0,
            )
            self.vxDLL.VortexSetInputReal(
                self.scene,
                'ML Interface'.encode('ascii'),
                'j6_vel_id'.encode('ascii'),
                0.0,
            )
            for i in range(self.pause_steps):
                self.vxDLL.VortexUpdateApplication()

    def _get_obs(self):
        self.joint_torques = self._readJtorque()
        self.joint_vel_real = self._readJvel()
        joint_poses = self._readJpos()
        joint_vel_id = []
        joint_vel_id.append(self.next_j_vel[0])
        joint_vel_id.append(self.next_j_vel[1])
        joint_vel_id.append(self.next_j_vel[2])

        print('Observation')
        print(f'joint_poses: {joint_poses}')
        print(f'\njoint_torques: {self.joint_torques}')
        print(f'\nJoint vels: {self.joint_vel_real}')
        print(f'\njoint_vel_id: {joint_vel_id}')

        # FIND OUT IF TORQUES ARE SURPASSING LIMITS #
        # if abs(self.joint_torques[0])>14.0:
        #   print("Joint 2 torque: {}".format(self.joint_torques[0]))
        # if abs(self.joint_torques[1])>5.0:
        #   print("Joint 4 torque: {}".format(self.joint_torques[1]))
        # if abs(self.joint_torques[2])>2.0:
        #   print("Joint 6 torque: {}".format(self.joint_torques[2]))

        return np.concatenate((self.joint_torques, self.joint_vel_real, joint_vel_id))

    def _readJpos(self):
        j2_pos_real_ = ctypes.c_double(0.0)
        j4_pos_real_ = ctypes.c_double(0.0)
        j6_pos_real_ = ctypes.c_double(0.0)
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_pos_real'.encode('ascii'),
            j2_pos_real_,
        )
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_pos_real'.encode('ascii'),
            j4_pos_real_,
        )
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_pos_real'.encode('ascii'),
            j6_pos_real_,
        )

        # j2_pos_real = j2_pos_real_.value+np.pi
        # j4_pos_real = j4_pos_real_.value+np.pi
        # j6_pos_real = j6_pos_real_.value+np.pi

        j2_pos_real = j2_pos_real_.value
        j4_pos_real = j4_pos_real_.value
        j6_pos_real = j6_pos_real_.value
        th_out = np.array([j2_pos_real, j4_pos_real, j6_pos_real])
        return th_out

    def _readJvel(self):
        j2_vel_real_ = ctypes.c_double(0.0)
        j4_vel_real_ = ctypes.c_double(0.0)
        j6_vel_real_ = ctypes.c_double(0.0)
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_vel_real'.encode('ascii'),
            j2_vel_real_,
        )
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_vel_real'.encode('ascii'),
            j4_vel_real_,
        )
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_vel_real'.encode('ascii'),
            j6_vel_real_,
        )
        return np.array([j2_vel_real_.value, j4_vel_real_.value, j6_vel_real_.value])

    def _readJtorque(self):
        j2_torque_ = ctypes.c_double(0.0)
        j4_torque_ = ctypes.c_double(0.0)
        j6_torque_ = ctypes.c_double(0.0)
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_torque'.encode('ascii'),
            j2_torque_,
        )
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_torque'.encode('ascii'),
            j4_torque_,
        )
        self.vxDLL.VortexGetOutputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_torque'.encode('ascii'),
            j6_torque_,
        )
        return np.array([j2_torque_.value, j4_torque_.value, j6_torque_.value])

    def _send_joint_target_vel(self, target_vels):
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j2_vel_id'.encode('ascii'),
            target_vels[0],
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j4_vel_id'.encode('ascii'),
            target_vels[1],
        )
        self.vxDLL.VortexSetInputReal(
            self.scene,
            'ML Interface'.encode('ascii'),
            'j6_vel_id'.encode('ascii'),
            target_vels[2],
        )

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

    def read_plug_force(self):
        plug_force = Vector3()
        self.vxDLL.VortexGetOutputVector3(
            self.scene,
            'ML Interface'.encode('ascii'),
            'plug_force'.encode('ascii'),
            plug_force,
        )
        return plug_force

    def read_plug_torque(self):
        plug_torque = Vector3()
        self.vxDLL.VortexGetOutputVector3(
            self.scene,
            'ML Interface'.encode('ascii'),
            'plug_torque'.encode('ascii'),
            plug_torque,
        )
        return plug_torque

    def get_insertion_depth(self):
        th_current = self._readJpos()
        x, z, rot = self._read_tips_pos_fk(th_current)
        return x, z, rot
