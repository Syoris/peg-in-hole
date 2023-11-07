
"""
Note that the API folder (/JACO-SDK/API/x64) must be added to the path environment in Windows
"""

from ctypes import *
import time
import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import h5py

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

# Function to DEFINE angular position command to be sent
## CAUTION: DO NOT USE THIS UNLESS YOU ARE CLOSE TO THE DESIRED POSITION ALREADY. ##
## ROBOT MAY MOVE QUICKLY OR IN UNEXPECTED WAYS. ##
def defCommandAngPos(command):
    point = (TrajectoryPoint)()
    point.InitStruct()
    point.Position.Type = ANGULAR_POSITION
    point.Position.Actuators.Actuator1 = command[0]
    point.Position.Actuators.Actuator2 = command[1]
    point.Position.Actuators.Actuator3 = command[2]
    point.Position.Actuators.Actuator4 = command[3]
    point.Position.Actuators.Actuator5 = command[4]
    point.Position.Actuators.Actuator6 = command[5]
    point.Position.Actuators.Actuator7 = command[6]
    return point

# Function to SEND angular velocity command
def sendCommand(point, n, sleep, phase, dll):
    t1 = 0.0
    for i in range(n):
        dll.SendBasicTrajectory(point)
        #resultAngVel = dll.GetAngularVelocity(dataAngularVel)
        #print("Ideal angular velocity of Joint 7: " + str(point.Position.Actuators.Actuator7) + " degrees/second.")
        #print("Actual angular velocity of Joint 7: " + str(dataAngularVel.Actuators.Actuator7) + " degrees/second.")
        #print("")
        time.sleep(sleep)
        # t2 = time.time()
        # t_dif = t2-t1
        # t1 = t2
    #print("Phase " + str(phase) + " is over.")

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

## EXAMPLES OF CALLING EACH FUNCTION ##

# Call API initialization function
# # This function initializes the API. It is the first function you call if you want the rest of the library.
# init_result = dll.InitAPI()
# print("Initialization's result: " + str(init_result))

# # Call API GetDevices
# # This function returns a list of devices accessible by this API.
# devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
# devicesCount = dll.GetDevices(devices, c_int(init_result))
# print("Number of connected devices: " + str(devicesCount))

# Call API GetAngularPosition
# This function returns the angular position of the robotical arm's end effector. Units are in degrees.
# COMMENT FROM ANYA FORESTELL: I believe this actually returns angular position of each actuator, not the end effector; tests need to be completed to verify this.
# dataAngularPos = (AngularPosition)()
# resultAngPos = dll.GetAngularPosition(dataAngularPos)
# print("Angular Position of Joint 7: " + str(dataAngularPos.Actuators.Actuator7) + " degrees.")

# Call API GetAngularVelocity
# This function gets the velocity of each actuator. Units are degrees / second.
# dataAngularVel = (AngularPosition)()
# resultAngVel = dll.GetAngularVelocity(dataAngularVel)
# print("Angular Velocity of Joint 7: " + str(dataAngularVel.Actuators.Actuator7) + " degrees/second.")

# Call API GetAngularCommand
# This function gets the angular command of all actuators. Units are degrees.
# dataAngularCom = (AngularPosition)()
# resultAngCom = dll.GetAngularCommand(dataAngularCom)
# print("Angular Command of Joint 7: " + str(dataAngularCom.Actuators.Actuator7) + " degrees.")

# Call API GetAngularForce
# This function returns the torque of each actuator. Unit is Newton meter [N * m].
# dataAngularFor = (AngularPosition)()
# resultAngFor = dll.GetAngularForce(dataAngularFor)
# print("Angular Force of Joint 7: " + str(dataAngularFor.Actuators.Actuator7) + " N*m.")

# Call API GetCartesianForce
# This function returns the cartesian force at the robotical arm's end effector. The translation unit is in Newtons and the orientation unit is Newton meters [N * m].
# dataCartesianFor = (CartesianPosition)()
# resultCartFor = dll.GetCartesianForce(dataCartesianFor)
# print("Cartesian Force in X-direction: " + str(dataCartesianFor.Coordinates.X) + " N.")
# print("Cartesian Torque along X-axis: " + str(dataCartesianFor.Coordinates.ThetaX) + " N*m.")

# Call API GetAngularCurrent
# This function returns the current that each actuator consumes on the main power supply. Unit is Amperes.
# dataAngularCur = (AngularPosition)()
# resultAngCur = dll.GetAngularCurrent(dataAngularCur)
# print("Current of Joint 7: " + str(dataAngularCur.Actuators.Actuator7) + " Amperes.")

# Call API SendBasicTrajectory (using our own defined functions for efficiency)
# This function sends a trajectory point (WITHOUT limitation) that will be added in the robotical arm's FIFO.
# x = 10 # angular velocity for a given joint
# n = 200 # number of steps to maintain a given velocity
# phase1 = defCommandAngVel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x]) # all joints are 0.0, except joint 7, which has velocity x degrees/second
# sendCommand(phase1, n, 0.005, 1, dll) # sending phase1 command for n steps, every 5ms

# Call API close function
# This function must called when your application stops using the API. It closes the USB link and the library properly.
# close_result = dll.CloseAPI()
# print("Closing's result: " + str(close_result))

### DEFINE ENVIRONMENT ###
    
class jaco2Env():
    # Initialize the environment and its variables. #
    def __init__(self):

        # Define action coefficient and reward weight.
        self.action_coeff = 0.01 # coefficient the action will be multiplied by
        self.reward_weight = 0.04 # we scale the reward using this coefficient
        self.collision = False # Controls when RL turns on - after collision becomes True
        #self.reward_min_threshold = -50.0 # Minimum threshold, below which simulation will end. NOT CURRENTLY USED NOR IMPLEMENTED IN THIS SCRIPT.

        # Define time and phase-related parameters.
        # self.h = 1.0/100.0                                      # Simulation time step (s)     
        self.h = 0.015                                          # Minimum frequency for sleep function is 0.01 seconds, plus 0.005 seconds for rest of program, so each timestep takes 0.015 seconds at minimum.
        self.t_pre_insert = 3.0                                # Time for pre-insertion (lowering from start to near block) (s)
        self.t_pause = 1.0                                      # Pause time (s) # NOT CURRENTLY USED
        self.t_insertion = 3.75                                 # Time for insertion (ensuring 250 steps per episode). (s)

        self.pause_steps = int(self.t_pause/self.h)             # Number of steps during Pause between phases
        self.pre_insert_steps = int(self.t_pre_insert/self.h)   # Number of steps for pre-insertion
        self.insertion_steps = int(self.t_insertion/self.h)     # Number of steps for insertion
        self.nStep = 0                                          # Step counter

        # self.xpos_hole = 0.529                                  # x position of the hole
        # self.xpos_hole = 0.542
        # self.xpos_hole = 0.5455
        # self.xpos_hole = 0.5415
        self.xpos_hole = 0.5484
        self.ypos_hole = -0.007                                 # y position of the hole

        self.max_misalign = 2.0                                 # maximum misalignment of insertion velocity
        self.min_misalign = 2.0                               # minimum misalignment of insertion velocity
        self.insertion_misalign = (np.pi/180.0) * (np.random.uniform(self.min_misalign, self.max_misalign))    # insertion velocity misalignment, negative and positive
        # self.insertion_misalign = (np.pi/180.0) * self.max_misalign    # insertion velocity misalignment, set max

        self.pre_insertz = 0.02                                 # z distance to be covered in pre-insertion phase
        self.insertz = 0.07                                     # z distance to be covered in insertion phase, though may change with actions

        # Define link lengths.
        self.L12 = 0.2755 # from base to joint 2, in metres
        self.L34 = 0.410  # from joint 2 to joint 4, in metres
        self.L56 = 0.3111 # from joint 4 to joint 6, in metres
        self.L78 = 0.2188 # from joint 6 to edge of End-Effector where peg is attached, in metres
        self.Ltip = 0.16  # from edge of End-Effector to tip of peg, in metres

        # Define joint starting points.
        self.j1_init = 90.0 # degrees # THIS IS 90 DEGREES DIFFERENT FROM VORTEX as we placed box at a different angle in Vortex than how robot can be clamped on table
        self.j2_init = 180.0 # degrees # THIS IS 180 DIFFERENT FROM VORTEX
        self.j3_init = 0.0 # degrees
        self.j4_init = 270.0 # degrees # THIS IS 180 DIFFERENT FROM VORTEX and we start already at 90 degrees from vertical
        self.j5_init = 0.0 # degrees
        self.j6_init = 180.0 # degrees # THIS IS 180 DIFFERENT FROM VORTEX
        self.j7_init = 0.0 # degrees
        self.j_init = np.array([self.j1_init, self.j2_init, self.j3_init, self.j4_init, self.j5_init, self.j6_init, self.j7_init])

        
        # self.x_set_list = []
        # with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/x_set_list.hdf5", 'r') as f:
        #     data = f['x_set_list']
        #     for x in data:
        #         self.x_set_list.append(x)
        
        # self.j2_2_set_list = []
        # self.j2_4_set_list = []
        # self.j4_2_set_list = []
        # self.j4_4_set_list = []
        # self.j6_2_set_list = []
        # self.j6_4_set_list = []
        # with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/calibration-contact-2degrees.hdf5", 'r') as f:
        #     j2_2 = f['j2_2_set']
        #     j4_2 = f['j4_2_set']
        #     j6_2 = f['j6_2_set']
        #     for i in range(len(j2_2)):
        #         self.j2_2_set_list.append(j2_2[i])
        #         self.j4_2_set_list.append(j4_2[i])
        #         self.j6_2_set_list.append(j6_2[i])
        # with h5py.File("C:/Users/Anya/Documents/GitHub/kinova-gen2-unjamming/3DOF/logs/calibration-contact-4degrees.hdf5", 'r') as f:
        #     j2_4 = f['j2_4_set']
        #     j4_4 = f['j4_4_set']
        #     j6_4 = f['j6_4_set']
        #     for i in range(len(j2_2)):
        #         self.j2_4_set_list.append(j2_4[i])
        #         self.j4_4_set_list.append(j4_4[i])
        #         self.j6_4_set_list.append(j6_4[i])

        # Define joint positions for pre-insertion position (close to block, just before insertion).
        # self.j2_preinsert = 119.38 # degrees                
        # self.j2_preinsert = 118.3 # degrees
        self.j2_preinsert = 117.49 # degrees                
        # self.j4_preinsert = 305.68 # degrees
        # self.j4_preinsert = 305.0 # degrees
        self.j4_preinsert = 304.67 # degrees
        # self.j6_preinsert = 276.29 # degrees
        # self.j6_preinsert = 277.2 # degrees
        self.j6_preinsert = 279.61
        self.j_preinsert = np.array([self.j2_preinsert, self.j4_preinsert, self.j6_preinsert])

        # Initialize joint velocity variables.
        self.next_j_vel = np.zeros(3)
        self.prev_j_vel = np.zeros(3)

        # Initialize the simulation to be incomplete.
        self.sim_completed = False


    # This function takes a step (used only in Insertion Phase). #
    def step(self, action, RL_used):
        ## Define insertion velocity misalignment. ##
        # self.insertion_misalign = (np.pi/180.0) * (np.random.uniform(self.min_misalign, self.max_misalign))    # insertion velocity misalignment, negative and positive
        # self.insertion_misalign = (np.pi/180.0) * self.max_misalign    # insertion velocity misalignment, set max
        ## Get the next ideal joint velocities and augment the actions to it. Send the augmented joint velocities to the robot. ##
        self.prev_j_vel = self.next_j_vel
        self.next_j_vel = self._get_ik_vels(self.insertz, self.nStep, self.insertion_steps, "insertion", RL_used) # in radians
        if self.collision == False:
            action = np.array([[0.0,0.0]])
        self.next_j_vel_send = np.array([self.next_j_vel[0] - self.action_coeff*action[0][0], self.next_j_vel[1] + self.action_coeff*action[0][1] - self.action_coeff*action[0][0], self.next_j_vel[2] + self.action_coeff*action[0][1]]) # in radians
        for i in range(len(self.next_j_vel_send)):
            self.next_j_vel_send[i] = self.next_j_vel_send[i] * 180.0/np.pi # changing from radians to degrees, so we can send to robot in right units (but training done in radians, as Vortex and action is in radians)

        self.next_j_vel_struct = defCommandAngVel([0.0, self.next_j_vel_send[0], 0.0, self.next_j_vel_send[1], 0.0, self.next_j_vel_send[2], 0.0])
        dll.SendBasicTrajectory(self.next_j_vel_struct)
        time.sleep(0.01)
        # sendCommand(self.next_j_vel_struct, 1, self.h, "None", dll)

        ## Get observations and reward. ##
        observations = self._get_obs()
        reward = self._get_reward()

        ## Check if simulation is done. ##
        self.nStep += 1
        if self.nStep >= self.insertion_steps:
            self.sim_completed = True

        return observations, reward, self.sim_completed, {}

    # This function resets the simulation AFTER at least one trial. #
    def reset(self, beginning):
        
        ## Get to preinsert point. ##
        # Move upwards 7cm, getting out of the hole. Only do this if it's NOT the beginning/only between trials, not before the first one.
        if not beginning:
            height1 = self.get_insertion_depth()[1]
            print("Height at start of leaving hole: {}".format(height1))
            for step in range(self.insertion_steps):
                self.next_j_vel_send = self._get_ik_vels(-self.insertz, step, self.insertion_steps, "pre_insert", False) # in radians
                for i in range(len(self.next_j_vel_send)):
                    self.next_j_vel_send[i] = self.next_j_vel_send[i] * 180.0/np.pi # changing from radians to degrees, so we can send to robot in right units (but training done in radians, as Vortex is in radians)
                command = np.array([0.0, self.next_j_vel_send[0], 0.0, self.next_j_vel_send[1], 0.0, self.next_j_vel_send[2], 0.0])
                self.next_j_vel_struct = defCommandAngVel(command)
                dll.SendBasicTrajectory(self.next_j_vel_struct)
                time.sleep(0.01)
                #sendCommand(self.next_j_vel_struct, 1, self.h, "None", dll)
            height2 = self.get_insertion_depth()[1]
            print("Height at end of leaving hole: {}".format(height2))

        ## End-effector moves to exact pre-insert position, near block's hole. ##
        # slowly bring close to pre-insert position
        order = [2,1,0]
        dataAngularPos = (AngularPosition)()
        resultAngPos = dll.GetAngularPosition(dataAngularPos)
        j = np.zeros(3) # FROM HERE WE ONLY USE PLANAR 3DOF JOINTS
        j[0] = dataAngularPos.Actuators.Actuator2
        j[1] = dataAngularPos.Actuators.Actuator4
        j[2] = dataAngularPos.Actuators.Actuator6
        for i in order: # This for loop will bring the peg down, close to the hole. We only use 3 joints at this point, for planar 3DOF motion.
            while abs(j[i]-self.j_preinsert[i]) > 0.01:
                j_sign = np.sign(j[i]-self.j_preinsert[i])
                x = -j_sign*2.5 # angular velocity for a given joint
                command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                command[2*i+1] = x
                phase3 = defCommandAngVel(command) # all joints are 0.0, except joint i, which has velocity x degrees/second
                dll.SendBasicTrajectory(phase3)
                time.sleep(0.01)
                resultAngPos = dll.GetAngularPosition(dataAngularPos)
                j = np.zeros(3)
                j[0] = dataAngularPos.Actuators.Actuator2
                j[1] = dataAngularPos.Actuators.Actuator4
                j[2] = dataAngularPos.Actuators.Actuator6
                print("Joint {} position is: {}".format(2*i+2, j[i]))
        # After bringing each joint close to the preinsert position at a low velocity, bring them to the exact position.
        # print("Bringing to exact preinsert point...")
        # command = np.array([self.j_init[0], self.j_preinsert[0], self.j_init[2], self.j_preinsert[1], self.j_init[4], self.j_preinsert[2], self.j_init[6]])
        # phase4 = defCommandAngPos(command)
        # dll.SendBasicTrajectory(phase4)
        # print("Done bringing to exact preinsert point!")

        self.nStep = 0
        self.sim_completed = False
        return self._get_obs()

    # This function brings the robot straight down. It is not precise, so it is to be used only to find the exact joint positions needed for a prefered position.
    def bringDown(self):
        # move down/up for a time
        height1 = self.get_insertion_depth()[1]
        print("Height at start: {}".format(height1))
        test_distance = 0.50          # in metres
        test_time = 25.0              # in seconds
        test_steps = int(test_time/self.h) # number of steps for this test
        for step in range(test_steps):
            self.next_j_vel_send = self._get_ik_vels(-test_distance, step, test_steps, "pre_insert", False) # in radians
            for i in range(len(self.next_j_vel_send)):
                self.next_j_vel_send[i] = self.next_j_vel_send[i] * 180.0/np.pi # changing from radians to degrees, so we can send to robot in right units (but training done in radians, as Vortex is in radians)
            command = np.array([0.0, self.next_j_vel_send[0], 0.0, self.next_j_vel_send[1], 0.0, self.next_j_vel_send[2], 0.0])
            self.next_j_vel_struct = defCommandAngVel(command)
            dll.SendBasicTrajectory(self.next_j_vel_struct)
            time.sleep(0.01)
        height2 = self.get_insertion_depth()[1]
        print("Height at end: {}".format(height2))
        joint_pos = self._readJpos()
        print(joint_pos)

    # This function returns next joint velocities in radians. #
    ## THIS IS FOR WHEN PEG ORIENTATION IS ALIGNED, BUT INSERTION DIRECTION IS MISALIGNED ##
    def _get_ik_vels(self, down_distance, cur_count, number_steps, phase, RL_used):
        th_current = self._readJpos()
        current_pos = self._read_tips_pos_fk(th_current)
        if phase == "pre_insert":
            x_set = current_pos[0] - self.xpos_hole
            x_vel = -x_set / ((number_steps - cur_count) * self.h) # distance to be covered in one timestep is the x_error divided by the number of steps left in this phase
            z_vel = down_distance / (number_steps * self.h) # distance to be covered in one timestep is total distance divided by total number of timesteps
            rot_set = current_pos[2] - np.pi
            rot_vel = -rot_set / ((number_steps - cur_count) * self.h)
        elif phase == "insertion":
            # vel = down_distance / (number_steps * self.h)
            # x_vel = vel * np.sin(self.insertion_misalign)
            # z_vel = vel * np.cos(self.insertion_misalign)
            # rot_set = current_pos[2] - np.pi
            # rot_vel = -rot_set / ((number_steps - cur_count) * self.h)

            # if RL_used == True:
            #     if cur_count < 50:
            #         vel = down_distance / (number_steps * self.h)
            #         x_set = current_pos[0] - (self.xpos_hole + down_distance*cur_count/number_steps * np.sin(self.insertion_misalign))
            #         x_vel = -x_set / (self.h)
            #         z_vel = vel * np.cos(self.insertion_misalign)
            #         rot_set = current_pos[2] - np.pi
            #         rot_vel = -rot_set / ((number_steps - cur_count) * self.h)
            #     else:
            #         vel = down_distance / (number_steps * self.h)
            #         x_vel = vel * np.sin(self.insertion_misalign)
            #         z_vel = vel * np.cos(self.insertion_misalign)
            #         rot_set = current_pos[2] - np.pi
            #         rot_vel = -rot_set / ((number_steps - cur_count) * self.h)
            # elif RL_used == False:
            #     vel = down_distance / (number_steps * self.h)
            #     x_set = current_pos[0] - (self.xpos_hole + down_distance*cur_count/number_steps * np.sin(self.insertion_misalign))
            #     x_vel = -x_set / (self.h)
            #     # x_vel = vel * np.sin(self.insertion_misalign)
            #     z_vel = vel * np.cos(self.insertion_misalign)
            #     rot_set = current_pos[2] - np.pi
            #     rot_vel = -rot_set / ((number_steps - cur_count) * self.h)
            # else:
            #     print("YOU MUST SPECIFY WHETHER RL IS USED IN GET_IK FUNCTION.")
            th_current = self._readJpos()
            tip_x_pos = self._read_tips_pos_fk(th_current)[0]
            print("TIP X:")
            print(tip_x_pos)
            print("X HIT WALL:")
            x_hit_wall = self.xpos_hole + 0.5*0.012 - 0.5*0.008
            print(x_hit_wall)
            print("")
            if cur_count>50 and tip_x_pos>x_hit_wall:
                self.collision=True
            if self.collision == False:
                vel = down_distance / (number_steps * self.h)
                x_set = current_pos[0] - (self.xpos_hole + down_distance*cur_count/number_steps * np.sin(self.insertion_misalign))
                x_vel = -x_set / (self.h)
                # z_set = current_pos[1] - (0.085 - down_distance*cur_count/number_steps * np.cos(self.insertion_misalign))
                # z_vel = z_set / self.h
                z_vel = vel * np.cos(self.insertion_misalign)
                rot_set = current_pos[2] - np.pi
                rot_vel = -rot_set / ((number_steps - cur_count) * self.h)
            else:
                vel = down_distance / (number_steps * self.h)
                x_vel = vel * np.sin(self.insertion_misalign)
                z_vel = vel * np.cos(self.insertion_misalign)
                rot_set = current_pos[2] - np.pi
                rot_vel = -rot_set / ((number_steps - cur_count) * self.h)
        else:
            print("STEP TYPES DOES NOT MATCH")
        
        next_vel = [x_vel, -z_vel, rot_vel]
        J = self._build_Jacobian(th_current)
        Jinv = np.linalg.inv(J)
        j_vel_next = np.dot(Jinv, next_vel)
        
        return j_vel_next # in radians/second
    
    # This function returns joint angles. Return values are in radians. #
    def _readJpos(self):
        dataAngularPos = (AngularPosition)()
        resultAngPos = dll.GetAngularPosition(dataAngularPos) # in degrees

        j2_pos_real = dataAngularPos.Actuators.Actuator2 * np.pi/180.0 - np.pi # in radians, subtracting pi due to robot's initial position starting at pi but Jacobian is based on starting at "zero"
        j4_pos_real = dataAngularPos.Actuators.Actuator4 * np.pi/180.0 - np.pi # in radians
        j6_pos_real = dataAngularPos.Actuators.Actuator6 * np.pi/180.0 - np.pi # in radians
        th_out = np.array([j2_pos_real, j4_pos_real, j6_pos_real]) # in radians

        return th_out
    
    # This function returns the tip of the peg's x position, z position, and angle of rotation about y axis (out of plane). #
    # Can only be used for planar 3DOF case, using Kinova Gen2 7DOF robot. #
    def _read_tips_pos_fk(self, th_current):
        q2 = th_current[0] # in radians
        q4 = th_current[1] # in radians
        q6 = th_current[2] # in radians
        
        current_tips_posx =  self.L34*np.sin(-q2) + self.L56*np.sin(-q2 +q4) + self.L78*np.sin(-q2 +q4 -q6) +self.Ltip*np.sin(-q2 +q4 -q6 +np.pi/2.0)
        current_tips_posz =  self.L12 + self.L34*np.cos(-q2) + self.L56*np.cos(-q2 +q4) + self.L78*np.cos(-q2 +q4 -q6) +self.Ltip*np.cos(-q2 +q4 -q6 +np.pi/2.0) + 0.079 # adding 0.115 to account for height of block and robot platform which are not taken into account in Vortex.
        current_tips_rot = -q2 + q4 - q6 + 90.0 * (np.pi / 180.0)

        return np.array([current_tips_posx, current_tips_posz, current_tips_rot])
    
    # This function constructs the Jacobian for the planar 3DOF version of the Kinova Gen2 7DOF arm (with joints 1, 3, 5, 7 frozen). #
    def _build_Jacobian(self, th_current):
        q2 = th_current[0] # in radians
        q4 = th_current[1] # in radians
        q6 = th_current[2] # in radians
        
        a_x = -self.L34*np.cos(-q2) - self.L56*np.cos(-q2 +q4) - self.L78*np.cos(-q2 +q4 -q6) - self.Ltip*np.cos(-q2 +q4 -q6 + np.pi/2.0)
        b_x = self.L56*np.cos(-q2 +q4) + self.L78*np.cos(-q2 +q4 -q6) + self.Ltip*np.cos(-q2 +q4 -q6 +np.pi/2.0)
        c_x = -self.L78*np.cos(-q2 +q4 -q6) - self.Ltip*np.cos(-q2 +q4 -q6 +np.pi/2.0)

        a_z = self.L34*np.sin(-q2) + self.L56*np.sin(-q2 +q4) + self.L78*np.sin(-q2 +q4 -q6) + self.Ltip*np.sin(-q2 +q4 -q6 + np.pi/2.0)
        b_z = -self.L56*np.sin(-q2 +q4) - self.L78*np.sin(-q2 +q4 -q6) - self.Ltip*np.sin(-q2 +q4 -q6 +np.pi/2.0)
        c_z = self.L78*np.sin(-q2 +q4 -q6) + self.Ltip*np.sin(-q2 +q4 -q6 +np.pi/2.0)

        J = [[a_x, b_x, c_x],[a_z, b_z, c_z],[-1.0, 1.0, -1.0]]

        return J
    
    # This function returns the observations (joint torques, actual joint velocities, ideal joint velocities). #
    def _get_obs(self):
        self.joint_torques = self._readJtorque() # in N*m
        self.joint_vel_real = self._readJvel() # in rad/second
        joint_vel_id = [] # in rad/second
        joint_vel_id.append(self.next_j_vel[0])
        joint_vel_id.append(self.next_j_vel[1])
        joint_vel_id.append(self.next_j_vel[2])
        # FIND OUT IF TORQUES ARE SURPASSING LIMITS #
        # if abs(self.joint_torques[0])>14.0:
        #   print("Joint 2 torque: {}".format(self.joint_torques[0]))
        # if abs(self.joint_torques[1])>5.0:
        #   print("Joint 4 torque: {}".format(self.joint_torques[1]))
        # if abs(self.joint_torques[2])>2.0:
        #   print("Joint 6 torque: {}".format(self.joint_torques[2]))

        return np.concatenate((self.joint_torques, self.joint_vel_real, joint_vel_id))
    
    # This function returns joint torques. #
    def _readJtorque(self):
        dataAngularFor = (AngularPosition)()
        resultAngFor = dll.GetAngularForce(dataAngularFor)
        #print("Angular Force of Joint 7: " + str(dataAngularFor.Actuators.Actuator7) + " N*m.")
        j2_torque_ = dataAngularFor.Actuators.Actuator2
        j4_torque_ = dataAngularFor.Actuators.Actuator4
        j6_torque_ = dataAngularFor.Actuators.Actuator6
        
        return np.array([j2_torque_, j4_torque_, j6_torque_])
    
    # This function returns actual joint velocities. #
    def _readJvel(self):
        dataAngularVel = (AngularPosition)()
        resultAngVel = dll.GetAngularVelocity(dataAngularVel) # in degrees/second
        #print("Angular Velocity of Joint 7: " + str(dataAngularVel.Actuators.Actuator7) + " degrees/second.")
        j2_vel_real_ = 2.0 * dataAngularVel.Actuators.Actuator2 * np.pi/180.0 # in rad/second, multiplied by 2 because robot is off by factor of 2...
        j4_vel_real_ = 2.0 *dataAngularVel.Actuators.Actuator4 * np.pi/180.0 # in rad/second, multiplied by 2 because robot is off by factor of 2...
        j6_vel_real_ = 2.0 *dataAngularVel.Actuators.Actuator6 * np.pi/180.0 # in rad/second, multiplied by 2 because robot is off by factor of 2...
     
        return np.array([j2_vel_real_, j4_vel_real_, j6_vel_real_]) # in rad/second
    
    # This function returns the reward based on power. #
    def _get_reward(self):
        j2_id = self.next_j_vel[0] # in rad/second
        j4_id = self.next_j_vel[1] # in rad/second
        j6_id = self.next_j_vel[2] # in rad/second

        #  reward = self.reward_weight*(-abs((shv_id-shv)*self.shoulder_torque.value)-abs((elv_id-elv)*self.elbow_torque.value)-abs((wrv_id-wrv)*self.wrist_torque.value))
        reward = self.reward_weight*( -abs((j2_id-self.joint_vel_real[0])*self.joint_torques[0])
                                    -abs((j4_id-self.joint_vel_real[1])*self.joint_torques[1])
                                    -abs((j6_id-self.joint_vel_real[2])*self.joint_torques[2]))
        return reward
    
    def read_plug_force(self):  
        dataCartesianFor = (CartesianPosition)()
        resultCartFor = dll.GetCartesianForce(dataCartesianFor)
        # print("Cartesian Force in X-direction: " + str(dataCartesianFor.Coordinates.X) + " N.")
        # print("Cartesian Torque along X-axis: " + str(dataCartesianFor.Coordinates.ThetaX) + " N*m.")

        force_x = dataCartesianFor.Coordinates.X
        force_y = dataCartesianFor.Coordinates.Y
        force_z = dataCartesianFor.Coordinates.Z
        
        torque_x = dataCartesianFor.Coordinates.ThetaX
        torque_y = dataCartesianFor.Coordinates.ThetaY
        torque_z = dataCartesianFor.Coordinates.ThetaZ
        
        force = np.array([force_x, force_y, force_z])
        torque = np.array([torque_x, torque_y, torque_z])
        return force, torque
  
    def get_insertion_depth(self):
        th_current = self._readJpos()
        tip_pos = self._read_tips_pos_fk(th_current)
        # print(tip_pos)
        return tip_pos
    
    # This function brings the robot from a random spot IN FREE SPACE [MAKE SURE THE ROBOT HAS SPACE TO MOVE] to the starting position near the block.
    def startPoint(self, sleep):
        dataAngularPos = (AngularPosition)()
        resultAngPos = dll.GetAngularPosition(dataAngularPos)
        order = [7,6,5,3,2,1,4]
        for i in order: # this loop will bring each joint close to the starting point, in L shape with joint4 at 90 degrees
            j = np.zeros(7)
            j[0] = dataAngularPos.Actuators.Actuator1
            j[1] = dataAngularPos.Actuators.Actuator2
            j[2] = dataAngularPos.Actuators.Actuator3
            j[3] = dataAngularPos.Actuators.Actuator4
            j[4] = dataAngularPos.Actuators.Actuator5
            j[5] = dataAngularPos.Actuators.Actuator6
            j[6] = dataAngularPos.Actuators.Actuator7

            while abs(j[i-1]-self.j_init[i-1]) > 0.01:
                j_sign = np.sign(j[i-1]-self.j_init[i-1])
                x = -j_sign*5.0 # angular velocity for a given joint
                n = 1 # number of steps to maintain a given velocity
                command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                command[i-1] = x
                phase1 = defCommandAngVel(command) # all joints are 0.0, except joint 7, which has velocity x degrees/second
                sendCommand(phase1, n, sleep, 1, dll) # sending phase1 command for n steps, every 5ms
                resultAngPos = dll.GetAngularPosition(dataAngularPos)
                j = np.zeros(7)
                j[0] = dataAngularPos.Actuators.Actuator1
                j[1] = dataAngularPos.Actuators.Actuator2
                j[2] = dataAngularPos.Actuators.Actuator3
                j[3] = dataAngularPos.Actuators.Actuator4
                j[4] = dataAngularPos.Actuators.Actuator5
                j[5] = dataAngularPos.Actuators.Actuator6
                j[6] = dataAngularPos.Actuators.Actuator7
                print("Joint {} position is: {}".format(i, j[i-1]))

        # After bringing each joint close to the starting position at a low velocity, bring them to the exact position.
        print("Bringing to exact initial point...")
        command = self.j_init
        phase2 = defCommandAngPos(command)
        dll.SendBasicTrajectory(phase2)
        print("Done bringing to exact initial point!")

        # slowly bring close to pre-insert position
        order = [2,1,0]
        dataAngularPos = (AngularPosition)()
        resultAngPos = dll.GetAngularPosition(dataAngularPos)
        j = np.zeros(3) # FROM HERE WE ONLY USE PLANAR 3DOF JOINTS
        j[0] = dataAngularPos.Actuators.Actuator2
        j[1] = dataAngularPos.Actuators.Actuator4
        j[2] = dataAngularPos.Actuators.Actuator6
        for i in order: # This for loop will bring the peg down, close to the hole. We only use 3 joints at this point, for planar 3DOF motion.
            while abs(j[i]-self.j_preinsert[i]) > 0.001:
                j_sign = np.sign(j[i]-self.j_preinsert[i])
                x = -j_sign*2.5 # angular velocity for a given joint
                command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                command[2*i+1] = x
                phase3 = defCommandAngVel(command) # all joints are 0.0, except joint i, which has velocity x degrees/second
                dll.SendBasicTrajectory(phase3)
                time.sleep(0.01)
                resultAngPos = dll.GetAngularPosition(dataAngularPos)
                j = np.zeros(3)
                j[0] = dataAngularPos.Actuators.Actuator2
                j[1] = dataAngularPos.Actuators.Actuator4
                j[2] = dataAngularPos.Actuators.Actuator6
                print("Joint {} position is: {}".format(2*i+2, j[i]))

        # After bringing each joint close to the preinsert position at a low velocity, bring them to the exact position.
        # print("Bringing to exact preinsert point...")
        # command = np.array([self.j_init[0], self.j_preinsert[0], self.j_init[2], self.j_preinsert[1], self.j_init[4], self.j_preinsert[2], self.j_init[6]])
        # phase4 = defCommandAngPos(command)
        # dll.SendBasicTrajectory(phase4)
        # print("Done bringing to exact preinsert point!")
        return self._get_obs()
    
    def prevstate(self):
        return self._get_obs()
