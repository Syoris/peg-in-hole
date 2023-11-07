#### SAFETY WARNING, PLEASE READ: This script will move the robot. Please ensure the robot begins from a safe position in free space. ####
#### This can be done either using the joystick or, before turning the robot on, manually moving the robot while it is off and supporting it in free space before turning it on. ####


"""
Note that the API folder (/JACO-SDK/API/x64) must be added to the path environment in Windows
"""

from ctypes import *
import time
import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from jaco2_env import jaco2Env # environment file should be saved in same directory
import h5py

session_name = "18September2023-test-nice-video" # this current session's name
train_session_name = "26July2023-run1" # name of the session from which trained RL model should be pulled from

env = jaco2Env() # The jaco2Env class is saved in the jaco_env.py file, which should be saved in the same directory. It contains the environment, inluding the step() function, reset() function, and other functions.
num_states = 9
num_actions = 2

j2_cmd_layer_api_path = "C:/Program Files (x86)/JACO-SDK/API/x64/CommandLayerWindows.dll"
j2_comm_layer_api_path = "C:/Program Files (x86)/JACO-SDK/API/x64/CommunicationLayerWindows.dll"

MAX_KINOVA_DEVICE = 20
SERIAL_LENGTH = 20

## DEFINE TYPES AND STRUCTURES NEEDED TO COMMUNICATE USING KINOVA API ##
# This is based on Kinova's API written in C++, adapted to work in Python using ctypes. #

# Define the POSITION_TYPE type
# That represents the type of a position. If used during a trajectory, the type of position
# will change the behaviour of the robot. For example if the position type is CARTESIAN_POSITION,
# then the robot's end effector will move to that position using the inverse kinematics. But
# if the type of position is CARTESIAN_VELOCITY then the robot will use the values as velocity command.
POSITION_TYPE = c_int
NOMOVEMENT_POSITION = POSITION_TYPE(0) # Used for initialisation.
CARTESIAN_POSITION = POSITION_TYPE(1)  # A cartesian position described by a translation X, Y, Z and an orientation ThetaX, thetaY and ThetaZ.
ANGULAR_POSITION = POSITION_TYPE(2)    # An angular position described by a value for each actuator.
RETRACTED = POSITION_TYPE(3)           # The robotic arm is in retracted mode. It may be anywhere between the HOME position and the RETRACTED position.
PREDEFINED1 = POSITION_TYPE(4)    	   # The robotic arm is moving to the pre defined position #1.
PREDEFINED2 = POSITION_TYPE(5)   	   # The robotic arm is moving to the pre defined position #2.
PREDEFINED3 = POSITION_TYPE(6)  	   # The robotic arm is moving to the pre defined position #3.
CARTESIAN_VELOCITY = POSITION_TYPE(7)  # A velocity vector used for velocity control.
ANGULAR_VELOCITY = POSITION_TYPE(8)    # Used for initialisation.
PREDEFINED4 = POSITION_TYPE(9)   	   # The robotic arm is moving to the pre defined position #4.
PREDEFINED5 = POSITION_TYPE(10)   	   # The robotic arm is moving to the pre defined position #5.
ANY_TRAJECTORY = POSITION_TYPE(11)     # Not used.
TIME_DELAY = POSITION_TYPE(12)         # The robotic arm is on time delay.

# Define the HAND_MODE type
# That indicates how the end effector will be used.
HAND_MODE = c_int
HAND_NOMOVEMENT = HAND_MODE(0) # Fingers will not move.
POSITION_MODE = HAND_MODE(1)   # Fingers will move using position control.
VELOCITY_MODE = HAND_MODE(2)   # Fingers will move using velocity control.
NO_FINGER = HAND_MODE(3)
ONE_FINGER = HAND_MODE(4)
TWO_FINGERS = HAND_MODE(5)
THREE_FINGERS = HAND_MODE(6)

# Define the KinovaDevice structure
# That is a device you can communicate with via this library.
class KinovaDevice(Structure):
    _fields_ = [
        ("SerialNumber", c_char * SERIAL_LENGTH), # The serial number of the device. If you are communicating with more than 1 device, this will be used to identify the devices.
        ("Model", c_char * SERIAL_LENGTH),        # The model of the device.
        ("VersionMajor", c_int),                  # Those variables represents the code version - Major.Minor.Release
        ("VersionMinor", c_int), 
        ("VersionRelease", c_int),
        ("DeviceType", c_int),                    # The type of the device.
        ("DeviceID", c_int)                       # This is a device ID used by the API. User should not use it.
    ]

# Define the AngularInfo structure, to be used in AngularPosition structure
# This data structure holds values in an angular (joint by joint) control context. As an example struct could contains position, temperature, torque, ...
class AngularInfo(Structure):
    _fields_ = [
        # As an example if the current control mode is angular position the unit will be degree but if the control mode is angular velocity
        # then the unit will be degree per second.
        ("Actuator1", c_float), 
        ("Actuator2", c_float),
        ("Actuator3", c_float),
        ("Actuator4", c_float),
        ("Actuator5", c_float),
        ("Actuator6", c_float),
        ("Actuator7", c_float)
    ]
    def InitStruct(self):
        self.Actuator1 = 0.0
        self.Actuator2 = 0.0
        self.Actuator3 = 0.0
        self.Actuator4 = 0.0
        self.Actuator5 = 0.0
        self.Actuator6 = 0.0
        self.Actuator7 = 0.0
    
# Define the FingersPosition strucre, to be used in AngularPosition structure
# This data structure holds the values of the robot's fingers. Units will depend on the context.
class FingersPosition(Structure):
    _fields_ = [
        ("Finger1", c_float),
        ("Finger2", c_float),
        ("Finger3", c_float)
    ]
    # This method will initialises all the values to 0:
    def InitStruct(self):
        self.Finger1 = 0.0
        self.Finger2 = 0.0
        self.Finger3 = 0.0

# Define the AngularPosition structure
# This data structure holds the values of an angular (actuators) position.
class AngularPosition(Structure):
    _fields_ = [
        ("Actuators", AngularInfo),
        ("Fingers", FingersPosition)
    ]
    # This method will initialises all the values to 0:
    def InitStruct(self):
        self.Actuators.InitStruct()
        self.Fingers.InitStruct()

# Define the CartesianInfo structure, to be used in UserPosition structure
# This data structure holds values in an cartesian control context.
class CartesianInfo(Structure):
    _fields_ = [
        # As an example if the current control mode is cartesian position the unit will be meters but if the control mode is cartesian velocity
	    # then the unit will be meters per second.
        ("X", c_float),      # Translation along X, Y, Z axes.
        ("Y", c_float),
        ("Z", c_float),
        # As an example if the current control mode is cartesian position the unit will be RAD but if the control mode is cartesian velocity
	    # then the unit will be RAD per second.
        ("ThetaX", c_float), # Orientation around X, Y, Z axes.
        ("ThetaY", c_float),
        ("ThetaZ", c_float)
    ]
    # This method will initialises all the values to 0:
    def InitStruct(self):
        self.X = 0.0
        self.Y = 0.0
        self.Z = 0.0
        self.ThetaX = 0.0
        self.ThetaY = 0.0
        self.ThetaZ = 0.0

# Define the CartesianPosition structure, to be used in determining Cartesian Force and Torque on End-effector.
# Coordinates holds the cartesian parts
    # of the position and Fingers holds contains the value of the fingers. As an example, if an instance
    # of the CartesianPosition is used in a cartesian velocity control context, the values in the struct
    # will be velocity.
# This data structure holds the values of a cartesian position.
# struct CartesianPosition KinovaTypes.h "Definition"
class CartesianPosition(Structure):
    _fields_ = [
        ("Coordinates", CartesianInfo), # This contains values regarding the cartesian information.(end effector).
        ("Fingers", FingersPosition)    # This contains value regarding the fingers.
    ]
    # This method will initialises all the values to 0:
    def InitStruct(self):
        self.Coordinates.InitStruct()
        self.Fingers.InitStruct()

# Define the UserPosition structure, to be used in TrajectoryPoint structure
# This data structure represents an abstract position built by a user. Depending on the control type the Cartesian information, the angular information or both will be used.
class UserPosition(Structure):
    _fields_ = [
        ("Type", POSITION_TYPE),              # figure out how to implement POSITION_TYPE. The type of this position.
        ("Delay", c_float),                   # This is used only if the type of position is TIME_DELAY. It represents the delay in second.
        ("CartesianPosition", CartesianInfo), # Cartesian information about this position.
        ("Actuators", AngularInfo),           # Angular information about this position.
        ("HandMode", HAND_MODE),              # figure out how to implement HAND_MODE. Mode of the gripper.
        ("Fingers", FingersPosition)          # Fingers information about this position.
    ]
    def InitStruct(self):
        self.Type = CARTESIAN_POSITION
        self.Delay = 0.0
        self.CartesianPosition.InitStruct()
        self.Actuators.InitStruct()
        self.HandMode = POSITION_MODE
        self.Fingers.InitStruct()
        

# Define the Limitation structure, to be used in TrajectoryPoint structure
# This data structure represents all limitation that can be applied to a control context.
# Depending on the context, units and behaviour can change. See each parameter for more informations.
class Limitation(Structure):
    _fields_ = [
        ("speedParameter1", c_float),        # In a cartesian context, this represents the translation velocity, but in an angular context, this represents the velocity of the actuators 1, 2 and 3.
        ("speedParameter2", c_float),        # In a cartesian context, this represents the orientation velocity, but in an angular context, this represents the velocity of the actuators 4, 5 and 6.
        ("speedParameter3", c_float),        # Not used for now.
        ("forceParameter1", c_float),        # Not used for now.
        ("forceParameter2", c_float),        # Not used for now.
        ("forceParameter3", c_float),        # Not used for now.
        ("accelerationParameter1", c_float), # Not used for now.
        ("accelerationParameter2", c_float), # Not used for now.
        ("accelerationParameter3", c_float)  # Not used for now.
    ]
    def InitStruct(self):
        self.speedParameter1 = 0.0
        self.speedParameter2 = 0.0
        self.speedParameter3 = 0.0
        self.forceParameter1 = 0.0
        self.forceParameter2 = 0.0
        self.forceParameter3 = 0.0
        self.accelerationParameter1 = 0.0
        self.accelerationParameter2 = 0.0
        self.accelerationParameter3 = 0.0

# Define the TrajectoryPoint structure
# This data structure represents a point of a trajectory. It contains the position a limitation that you can applied (sic).

class TrajectoryPoint(Structure):
    _fields_ = [
        ("Position", UserPosition),   # Position information that described this trajectory point.
        ("LimitationsActive", c_int), # A flag that indicates if the limitation are active or not (1 is active 0 is not).
        ("SynchroType", c_int),       # A flag that indicates if the tracjetory's synchronization is active. (1 is active 0 is not).
        ("Limitations", Limitation)   # Limitation applied to this point if the limitation flag is active.
    ]
    def InitStruct(self):
        self.Position.InitStruct()
        self.LimitationsActive = 0
        self.SynchroType = 0
        self.Limitations.InitStruct()

## DEFINE FUNCTIONS FOR EFFICIENCY ##
# These are not originally part of Kinova's API. They were defined by Anya Forestell for ease-of-use. #

# Function to DEFINE angular velocity command to be sent
def defCommandAngVel(command):
    point = (TrajectoryPoint)()
    point.InitStruct()
    point.Position.Type = ANGULAR_VELOCITY
    point.Position.Actuators.Actuator1 = command[0]
    point.Position.Actuators.Actuator2 = command[1]
    point.Position.Actuators.Actuator3 = command[2]
    point.Position.Actuators.Actuator4 = command[3]
    point.Position.Actuators.Actuator5 = command[4]
    point.Position.Actuators.Actuator6 = command[5]
    point.Position.Actuators.Actuator7 = command[6]
    return point

# Function to SEND angular velocity command
def sendCommand(point, n, sleep, phase):
    t1 = 0.0
    for i in range(n):
        dll.SendBasicTrajectory(point)
        resultAngVel = dll.GetAngularVelocity(dataAngularVel)
        print("Ideal angular velocity of Joint 7: " + str(point.Position.Actuators.Actuator7) + " degrees/second.")
        print("Actual angular velocity of Joint 7: " + str(dataAngularVel.Actuators.Actuator7) + " degrees/second.")
        print("")
        time.sleep(sleep)
        # t2 = time.time()
        # t_dif = t2-t1
        # t1 = t2
    print("Phase " + str(phase) + " is over.")

# Function to create actor network
def get_actor():
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(32, activation="relu")(inputs)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh")(out)

    model = tf.keras.Model(inputs, outputs)
    return model

# Function to create critic network
def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    # Action as input
    action_input = layers.Input(shape=(num_actions))

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(32, activation="relu")(concat)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)
    out = layers.Dense(32, activation="relu")(out)

    outputs = layers.Dense(1, activation='linear')(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

# Function to output test policy (policy based on trained model)
def test_policy(state):
    sampled_actions = tf.squeeze(target_actor(state))

    # Turning it into numpy values?
    sampled_actions = sampled_actions.numpy()
   
    # We make sure action is within bounds
    upper_bound = np.array([1.0, 1.0])
    lower_bound = np.array([-1.0, -1.0])
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)] # joint velocities in radians

# Function to create h5py dataset files and save the first trial's data.
def create_file(file, data_label, data):
    data = np.asanyarray(data)
    result = file.create_dataset(data_label, data=data, maxshape=(None,) + data.shape[1:])
    return result

# Function to append further trials' data to existing datasets in h5py file.
def append_to_dataset(dataset, data):
    data = np.asanyarray(data)
    dataset.resize(len(dataset) + len(data), axis=0)
    dataset[-len(data):] = data

## LOAD DLL ##
dll = CDLL(j2_cmd_layer_api_path)

## DEFINE FUNCTION OUTPUT TYPES AND ARGUMENT TYPES ##

# Define function's output type
dll.InitAPI.restype = c_int
dll.CloseAPI.restype = c_int

dll.GetDevices.restype = c_int
dll.GetAngularPosition.restype = c_int
dll.GetAngularVelocity.restype = c_int
dll.GetAngularCommand.restype = c_int
dll.GetAngularForce.restype = c_int
dll.GetAngularCurrent.restype = c_int
dll.GetCartesianForce.restype = c_int

dll.SendBasicTrajectory.restype = c_int

# You need to specify the argument types for the function
dll.GetDevices.argtypes = [POINTER(KinovaDevice), POINTER(c_int)]
dll.GetAngularPosition.argtypes = [POINTER(AngularPosition)]
dll.GetAngularVelocity.argtypes = [POINTER(AngularPosition)]
dll.GetAngularCommand.argtypes = [POINTER(AngularPosition)]
dll.GetAngularForce.argtypes = [POINTER(AngularPosition)]
dll.GetAngularCurrent.argtypes = [POINTER(AngularPosition)]
dll.GetCartesianForce.argtypes = [POINTER(CartesianPosition)]

dll.SendBasicTrajectory.argtypes = [POINTER(TrajectoryPoint)]

## TRAINING HYPERPARAMETERS ##
std_dev = 0.2

target_actor = get_actor()
target_critic = get_critic()

target_actor.load_weights("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/RL_setup/models/arm_target_actor_{}.h5".format(train_session_name))
target_critic.load_weights("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/RL_setup/models/arm_target_critic_{}.h5".format(train_session_name))

total_episodes = 1 # We will be doing only one trial at a time for safety reasons.
# If the robot resets outside acceptable ranges of error, which could lead to the peg making contact with the block OUTSIDE the hole and not inserting, the simulation should be IMMEDIATELY STOPPED and restarted.

# Discount factor for future rewards
gamma = 0.99

## INITIALIZING ROBOT ##

# Call API initialization function
# This function initializes the API. It is the first function you call if you want the rest of the library.
init_result = dll.InitAPI()
print("Initialization's result: " + str(init_result))

# Call API GetDevices
# This function returns a list of devices accessible by this API.
devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
devicesCount = dll.GetDevices(devices, c_int(init_result))
print("Number of connected devices: " + str(devicesCount))

prev_state = env.reset(True) # brings the robot to pre-insert position. "True" means it is the first trial, so it will not move upwards first (which is only used to reset once already in the hole).

# print("Tip position is:")
# env.get_insertion_depth()

RL_used = False
trial = 1

if RL_used == True:
    ## TESTING WITH RL ##
    ## Commented out for the time being, as we are only doing trials without RL. ##

    print("PAUSING BEFORE RL")
    time.sleep(7)

    # To store reward history of each episode
    ep_reward_list = []
    end_depth_list = []
    ep_misalign_list = []

    for ep in range(total_episodes):
        # To store individual reward/force/torque for each step over one episode
        ind_reward_list = []
        ind_force_list = []
        ind_torque_list = []
        ind_j2_torque_list = []
        ind_j4_torque_list = []
        ind_j6_torque_list = []
        ind_height_list = []
        ind_action_list = []
        observation_list = []
        ind_x_list = []
        ind_rot_list = []

        episodic_reward = 0

        count = 0
        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.        

            # if ep == total_episodes-1:
            #     env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = test_policy(tf_prev_state) # test WITH RL

            # Recieve state and reward from environment. Also get force/torque at end-effector, torque for each joint, insertion height at this point.
            state, reward, done, info = env.step(action, RL_used)

            force, torque = env.read_plug_force()
            force_norm = np.sqrt(force[0]**2.0 + force[1]**2.0 + force[2]**2.0)
            torque_norm = np.sqrt(torque[0]**2.0 + torque[1]**2.0+torque[2]**2.0)

            j2_torque = state[0]
            j4_torque = state[1]
            j6_torque = state[2]

            height = env.get_insertion_depth()[1]
            tip_x_position = env.get_insertion_depth()[0]
            tip_rotation = env.get_insertion_depth()[2]

            # ADD READ CURRENT?
            ind_reward_list.append(reward)
            ind_force_list.append(force)
            ind_torque_list.append(torque)
            ind_j2_torque_list.append(j2_torque)
            ind_j4_torque_list.append(j4_torque)
            ind_j6_torque_list.append(j6_torque)
            ind_height_list.append(height)
            ind_action_list.append(action)
            observation_list.append(state)
            ind_x_list.append(tip_x_position)
            ind_rot_list.append(tip_rotation)

            episodic_reward += reward
            
            count += 1

            # End this episode when `done` is True
            if done:
                break

            prev_state = state
        
        ep_reward_list.append(episodic_reward)
        ep_misalign_list.append(env.insertion_misalign)
        print("Avg reward per step over episode: " + str(episodic_reward/count))
        print("Total reward over episode: " + str(episodic_reward))
        
        insert_depth = env.get_insertion_depth()
        print("Episode * {} * Insertion Height is ==> {}".format(ep, insert_depth))
        print("")
        end_depth_list.append(insert_depth)
        # RESET, bringing peg out of hole and back to exact preinsert position.
        # print("PAUSING AFTER RL")
        time.sleep(2)
        
        prev_state = env.reset(False)

    # Save the data #
    if trial == 1:
        with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/RL_log_{}.hdf5".format(session_name), 'w') as f:
            create_file(f, "ind reward", np.array([ind_reward_list]))
            create_file(f, "reward", np.array([ep_reward_list]))
            create_file(f, "force", np.array([ind_force_list]))
            create_file(f, "torque", np.array([ind_torque_list]))
            create_file(f, "torque_j2", np.array([ind_j2_torque_list]))
            create_file(f, "torque_j4", np.array([ind_j4_torque_list]))
            create_file(f, "torque_j6", np.array([ind_j6_torque_list]))
            create_file(f, "ind height", np.array([ind_height_list]))
            create_file(f, "height", np.array([end_depth_list]))
            create_file(f, "ind action", np.array([ind_action_list]))
            create_file(f, "misalignment velocity", np.array([ep_misalign_list]))
            create_file(f, "state", np.array([observation_list]))
            create_file(f, "xpos", np.array([ind_x_list]))
            create_file(f, "rotation", np.array([ind_rot_list]))

    else:
        with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/RL_log_{}.hdf5".format(session_name), 'a') as f:
            append_to_dataset(f["ind reward"], np.array([ind_reward_list]))
            append_to_dataset(f["reward"], np.array([ep_reward_list]))
            append_to_dataset(f["force"], np.array([ind_force_list]))
            append_to_dataset(f["torque"], np.array([ind_torque_list]))
            append_to_dataset(f["torque_j2"], np.array([ind_j2_torque_list]))
            append_to_dataset(f["torque_j4"], np.array([ind_j4_torque_list]))
            append_to_dataset(f["torque_j6"], np.array([ind_j6_torque_list]))
            append_to_dataset(f["ind height"], np.array([ind_height_list]))
            append_to_dataset(f["height"], np.array([end_depth_list]))
            append_to_dataset(f["ind action"], np.array([ind_action_list]))
            append_to_dataset(f["misalignment velocity"], np.array([ep_misalign_list]))
            append_to_dataset(f["state"], np.array([observation_list]))
            append_to_dataset(f["xpos"], np.array([ind_x_list]))
            append_to_dataset(f["rotation"], np.array([ind_rot_list]))

# print("PAUSING BETWEEN RL AND NO RL")
elif RL_used == False:
    print("PAUSING BEFORE NO RL")
    time.sleep(7) # 7 second pause
    print("RESUMING, TESTING WITHOUT RL")

    ## TESTING WITHOUT RL ##

    # To store reward history of each episode
    ep_reward_list_noRL = []
    end_depth_list_noRL = []
    ep_misalign_list_noRL = []

    for ep in range(total_episodes):

        # prev_state = env.reset()
        # prev_state = prev_state[0]
        # To store individual reward/force/torque for each step over one episode
        ind_reward_list_noRL = []
        ind_force_list_noRL = []
        ind_torque_list_noRL = []
        ind_j2_torque_list_noRL = []
        ind_j4_torque_list_noRL = []
        ind_j6_torque_list_noRL = []
        ind_height_list_noRL = []
        ind_action_list_noRL = []
        observation_list_noRL = []
        ind_x_list_noRL = []
        ind_rot_list_noRL = []
        
        episodic_reward_noRL = 0

        count = 0
        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.        

            # if ep == total_episodes-1:
            #     env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = np.array([[0.0,0.0]]) # test WITHOUT RL

            # Recieve state and reward from environment.
            # state, reward, done, info = env.step(action, RL_used)
            state, reward, done, info = env.step(action, True)

            force, torque = env.read_plug_force()
            force_norm = np.sqrt(force[0]**2.0 + force[1]**2.0 + force[2]**2.0)
            torque_norm = np.sqrt(torque[0]**2.0 + torque[1]**2.0+torque[2]**2.0)

            j2_torque = state[0]
            j4_torque = state[1]
            j6_torque = state[2]
            
            height = env.get_insertion_depth()[1]
            tip_x_position = env.get_insertion_depth()[0]
            tip_rotation = env.get_insertion_depth()[2]

            ind_reward_list_noRL.append(reward)
            ind_force_list_noRL.append(force)
            ind_torque_list_noRL.append(torque)
            ind_j2_torque_list_noRL.append(j2_torque)
            ind_j4_torque_list_noRL.append(j4_torque)
            ind_j6_torque_list_noRL.append(j6_torque)
            ind_height_list_noRL.append(height)
            ind_action_list_noRL.append(action)
            observation_list_noRL.append(state)
            ind_x_list_noRL.append(tip_x_position)
            ind_rot_list_noRL.append(tip_rotation)

            episodic_reward_noRL += reward
            
            count += 1

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list_noRL.append(episodic_reward_noRL)
        ep_misalign_list_noRL.append(env.insertion_misalign)

        print("Avg reward per step over episode, _noRL: " + str(episodic_reward_noRL/count))
        print("Total reward over episode, _noRL: " + str(episodic_reward_noRL))
        
        insert_depth = env.get_insertion_depth()
        print("Episode * {} * Insertion Height is ==> {}".format(ep, insert_depth))
        print("")
        end_depth_list_noRL.append(insert_depth)
        # RESET, bringing peg out of hole and back to exact preinsert position.
        time.sleep(2)
        prev_state = env.reset(False)

    # Save the data #
    if trial == 1: # creates datasets and stores data of first trial
        with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/noRL_log_{}.hdf5".format(session_name), 'w') as f:
            create_file(f, "ind reward", np.array([ind_reward_list_noRL]))   # creates dataset and stores reward at each step of first trial (250 datapoints per trial)
            create_file(f, "reward", np.array([ep_reward_list_noRL]))        # creates dataset and stores total summed reward over whole first trial (1 datapoint per trial)
            create_file(f, "force", np.array([ind_force_list_noRL]))         # creates dataset and stores x-y-z forces on End Effector for each step of first trial (250x3 datapoints per trial)
            create_file(f, "torque", np.array([ind_torque_list_noRL]))       # creates dataset and stores x-y-z torques on End Effector for each step of first trial (250x3 datapoints per trial)
            create_file(f, "torque_j2", np.array([ind_j2_torque_list_noRL])) # creates dataset and stores j2 torque for each step of first trial (250 datapoints per trial)
            create_file(f, "torque_j4", np.array([ind_j4_torque_list_noRL])) # creates dataset and stores j4 torque for each step of first trial (250 datapoints per trial)
            create_file(f, "torque_j6", np.array([ind_j6_torque_list_noRL])) # creates dataset and stores j6 torque for each step of first trial (250 datapoints per trial)
            create_file(f, "ind height", np.array([ind_height_list_noRL]))   # creates dataset and stores height of tip of peg for each step of first trial (250 datapoints per trial)
            create_file(f, "height", np.array([end_depth_list_noRL]))        # creates dataset and stores the height of tip of peg at the end of first trial (1 datapoint per trial)
            create_file(f, "ind action", np.array([ind_action_list_noRL]))   # creates dataset and stores action at each step of first trial (250x2 datapoints per trial)
            create_file(f, "misalignment velocity", np.array([ep_misalign_list_noRL]))
            create_file(f, "state", np.array([observation_list_noRL]))
            create_file(f, "xpos", np.array([ind_x_list_noRL]))
            create_file(f, "rotation", np.array([ind_rot_list_noRL]))

    else: # as dataset is created in first trial, trial 2+ just append to datasets and store new data
        with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/noRL_log_{}.hdf5".format(session_name), 'a') as f:
            append_to_dataset(f["ind reward"], np.array([ind_reward_list_noRL]))
            append_to_dataset(f["reward"], np.array([ep_reward_list_noRL]))
            append_to_dataset(f["force"], np.array([ind_force_list_noRL]))
            append_to_dataset(f["torque"], np.array([ind_torque_list_noRL]))
            append_to_dataset(f["torque_j2"], np.array([ind_j2_torque_list_noRL]))
            append_to_dataset(f["torque_j4"], np.array([ind_j4_torque_list_noRL]))
            append_to_dataset(f["torque_j6"], np.array([ind_j6_torque_list_noRL]))
            append_to_dataset(f["ind height"], np.array([ind_height_list_noRL]))
            append_to_dataset(f["height"], np.array([end_depth_list_noRL]))
            append_to_dataset(f["ind action"], np.array([ind_action_list_noRL]))
            append_to_dataset(f["misalignment velocity"], np.array([ep_misalign_list_noRL]))
            append_to_dataset(f["state"], np.array([observation_list_noRL]))
            append_to_dataset(f["xpos"], np.array([ind_x_list_noRL]))
            append_to_dataset(f["rotation"], np.array([ind_rot_list_noRL]))

else:
    print("YOU MUST SPECIFY WHETHER RL IS USED.")

# x_set_list = env.x_set_list
# with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/x_set_list.hdf5", 'w') as f:
#     create_file(f, "x_set_list", x_set_list)
            
# j2_2_list = env.j2_2_set_list
# j2_4_list = env.j2_4_set_list
# j4_2_list = env.j4_2_set_list
# j4_4_list = env.j4_4_set_list
# j6_2_list = env.j6_2_set_list
# j6_4_list = env.j6_4_set_list
# with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/calibration-contact-4degrees.hdf5", 'w') as f:
#     # create_file(f, "j2_2_set", j2_2_list)
#     create_file(f, "j2_4_set", j2_4_list)
#     # create_file(f, "j4_2_set", j4_2_list)
#     create_file(f, "j4_4_set", j4_4_list)
#     # create_file(f, "j6_2_set", j6_2_list)
#     create_file(f, "j6_4_set", j6_4_list)

## PLOT GRAPHS ##
# The plan is to separate this into another script that processes the data and makes plots. #

# action_coeff = env.action_coeff
# max_misalign = env.max_misalign
# #reward_min_threshold = env.reward_min_threshold
# reward_min_threshold = "None"

# # Episodes versus Episode Rewards
# plt.figure(figsize=(14,7))
# plt.plot(ep_reward_list, label="With RL")
# plt.plot(ep_reward_list_noRL, label="Without RL")
# plt.xlabel("Episode")
# plt.ylabel("Episodic Reward: sum of reward of all steps in one episode")
# plt.legend()
# # plt.title("Ind. Reward for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_reward.png".format(session_name))
# plt.close()

# # Episodes versus Step Rewards
# plt.figure(figsize=(14,7))
# plt.plot(ind_reward_list, label="With RL")
# plt.plot(ind_reward_list_noRL, label="Without RL")
# plt.xlabel("Step")
# plt.ylabel("Step Reward: reward from each step")
# plt.legend()
# # plt.title("Ind. Reward for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_ind_reward.png".format(session_name))
# plt.close()

# # Episodes versus norm forces
# plt.figure(figsize=(14,7))
# plt.plot(ind_force_list, label="With RL")
# plt.plot(ind_force_list_noRL, label="Without RL")
# plt.xlabel("Step")
# plt.ylabel("Step Force (norm of x-y-z)")
# plt.legend()
# #plt.title("Force for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_ind_force.png".format(session_name))
# plt.close()

# plt.figure(figsize=(14,7))
# plt.plot(ind_torque_list, label="With RL")
# plt.plot(ind_torque_list_noRL, label="Without RL")
# plt.xlabel("Step")
# plt.ylabel("Step Torque (norm of x-y-z)")
# plt.legend()
# #plt.title("Torque for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_ind_torque.png".format(session_name))
# plt.close()

# plt.figure(figsize=(14,7))
# plt.plot(ind_j2_torque_list, label="With RL")
# plt.plot(ind_j2_torque_list_noRL, label="Without RL")
# plt.xlabel("Step")
# plt.ylabel("Step Torque for Joint 2")
# plt.legend()
# #plt.title("Torque for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_ind_j2_torque.png".format(session_name))
# plt.close()

# plt.figure(figsize=(14,7))
# plt.plot(ind_j4_torque_list, label="With RL")
# plt.plot(ind_j4_torque_list_noRL, label="Without RL")
# plt.xlabel("Step")
# plt.ylabel("Step Torque for Joint 4")
# plt.legend()
# #plt.title("Torque for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_ind_j4_torque.png".format(session_name))
# plt.close()

# plt.figure(figsize=(14,7))
# plt.plot(ind_j6_torque_list, label="With RL")
# plt.plot(ind_j6_torque_list_noRL, label="Without RL")
# plt.xlabel("Step")
# plt.ylabel("Step Torque for Joint 6")
# plt.legend()
# #plt.title("Torque for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_ind_j6_torque.png".format(session_name))
# plt.close()

# # Insertion depth of episode
# plt.figure(figsize=(14,7))
# plt.plot(end_depth_list, label="With RL")
# plt.plot(end_depth_list_noRL, label="Without RL")
# plt.xlabel("Episode")
# plt.ylabel("Height of tip at end of episode (m)")
# plt.legend()
# #plt.title("Avg Height for action_coeff = {}, misalignment = {} degrees, reward_min_threshold = {}.".format(action_coeff, max_misalign, reward_min_threshold))
# #plt.show()
# plt.savefig("C:/Users/Anya/Documents/McGill/Plots RL/WORKING VERSION/physical-robot/{}_height.png".format(session_name))
# plt.close()

## CLOSE API ##

# Call API close function
# This function must called when your application stops using the API. It closes the USB link and the library properly.
close_result = dll.CloseAPI()
print("Closing's result: " + str(close_result))
