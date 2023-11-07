
"""
Note that the API folder (/JACO-SDK/API/x64) must be added to the path environment in Windows
"""

from ctypes import *
import time
import numpy as np

j2_cmd_layer_api_path = "C:/Program Files (x86)/JACO-SDK/API/x64/CommandLayerWindows.dll"
j2_comm_layer_api_path = "C:/Program Files (x86)/JACO-SDK/API/x64/CommunicationLayerWindows.dll"

MAX_KINOVA_DEVICE = 20
SERIAL_LENGTH = 20

# Define joint starting points
j1_init = 90.0 # degrees # THIS IS 90 DEGREES DIFFERENT FROM VORTEX as we placed box at a different angle in Vortex than how robot can be clamped on table
j2_init = 184.0 # degrees
j3_init = 0.0 # degrees
j4_init = 270.0 # degrees
j5_init = 0.0 # degrees
j6_init = 180.0 # degrees
j7_init = 0.0 # degrees
j_init = np.array([j1_init, j2_init, j3_init, j4_init, j5_init, j6_init, j7_init])

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

## Define functions for efficiency ##

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
        #print("Ideal angular velocity of Joint 7: " + str(point.Position.Actuators.Actuator7) + " degrees/second.")
        #print("Actual angular velocity of Joint 7: " + str(dataAngularVel.Actuators.Actuator7) + " degrees/second.")
        #print("")
        time.sleep(sleep)
        # t2 = time.time()
        # t_dif = t2-t1
        # t1 = t2
    #print("Phase " + str(phase) + " is over.")

# Function to get robot to starting point
def startPoint(sleep):
    dataAngularPos = (AngularPosition)()
    resultAngPos = dll.GetAngularPosition(dataAngularPos)
    order = [7,6,5,3,2,1,4]
    for i in order:
        j = np.zeros(7)
        j[0] = dataAngularPos.Actuators.Actuator1
        j[1] = dataAngularPos.Actuators.Actuator2
        j[2] = dataAngularPos.Actuators.Actuator3
        j[3] = dataAngularPos.Actuators.Actuator4
        j[4] = dataAngularPos.Actuators.Actuator5
        j[5] = dataAngularPos.Actuators.Actuator6
        j[6] = dataAngularPos.Actuators.Actuator7

        while abs(j[i-1]-j_init[i-1]) > 0.01:
            j_sign = np.sign(j[i-1]-j_init[i-1])
            x = -j_sign*5.0 # angular velocity for a given joint
            n = 1 # number of steps to maintain a given velocity
            command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            command[i-1] = x
            phase1 = defCommandAngVel(command) # all joints are 0.0, except joint 7, which has velocity x degrees/second
            sendCommand(phase1, n, sleep, 1) # sending phase1 command for n steps, every 5ms
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

# Load dll
dll = CDLL(j2_cmd_layer_api_path)

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

# Call API initialization function
# This function initializes the API. It is the first function you call if you want the rest of the library.
init_result = dll.InitAPI()
print("Initialization's result: " + str(init_result))

# Call API GetDevices
# This function returns a list of devices accessible by this API.
devices = (KinovaDevice * MAX_KINOVA_DEVICE)()
devicesCount = dll.GetDevices(devices, c_int(init_result))
print("Number of connected devices: " + str(devicesCount))

# Call API GetAngularPosition
# This function returns the angular position of the robotical arm's end effector. Units are in degrees.
dataAngularPos = (AngularPosition)()
resultAngPos = dll.GetAngularPosition(dataAngularPos)
print("Angular Position of Joint 6: " + str(dataAngularPos.Actuators.Actuator6) + " degrees.")

# Call API GetAngularVelocity
# This function gets the velocity of each actuator. Units are degrees / second.
dataAngularVel = (AngularPosition)()
resultAngVel = dll.GetAngularVelocity(dataAngularVel)
print("Angular Velocity of Joint 7: " + str(dataAngularVel.Actuators.Actuator7) + " degrees/second.")

# Call API GetAngularCommand
# This function gets the angular command of all actuators. Units are degrees.
dataAngularCom = (AngularPosition)()
resultAngCom = dll.GetAngularCommand(dataAngularCom)
print("Angular Command of Joint 7: " + str(dataAngularCom.Actuators.Actuator7) + " degrees.")

# Call API GetAngularForce
# This function returns the torque of each actuator. Unit is Newton meter [N * m].
dataAngularFor = (AngularPosition)()
resultAngFor = dll.GetAngularForce(dataAngularFor)
print("Angular Force of Joint 7: " + str(dataAngularFor.Actuators.Actuator7) + " N*m.")

# Call API GetCartesianForce
# This function returns the cartesian force at the robotical arm's end effector. The translation unit is in Newtons and the orientation unit is Newton meters [N * m].
dataCartesianFor = (CartesianPosition)()
resultCartFor = dll.GetCartesianForce(dataCartesianFor)
print("Cartesian Force in X-direction: " + str(dataCartesianFor.Coordinates.X) + " N.")
print("Cartesian Torque along X-axis: " + str(dataCartesianFor.Coordinates.ThetaX) + " N*m.")

# Call API GetAngularCurrent
# This function returns the current that each actuator consumes on the main power supply. Unit is Amperes.
dataAngularCur = (AngularPosition)()
resultAngCur = dll.GetAngularCurrent(dataAngularCur)
print("Current of Joint 7: " + str(dataAngularCur.Actuators.Actuator7) + " Amperes.")

# Call API SendBasicTrajectory (using our own defined functions for efficiency)
# This function sends a trajectory point (WITHOUT limitation) that will be added in the robotical arm's FIFO.
# x = 10.0 # angular velocity for a given joint
# n = 200 # number of steps to maintain a given velocity
# phase1 = defCommandAngVel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x]) # all joints are 0.0, except joint 7, which has velocity x degrees/second
# sendCommand(phase1, n, 0.009, 1) # sending phase1 command for n steps, every 5ms

# Test bringing joints to starting point. Make it a function where you can just say what joint position you want WITH a velocity?

startPoint(0.005)

# Bring End-effector downwards.

# Call API close function
# This function must called when your application stops using the API. It closes the USB link and the library properly.
close_result = dll.CloseAPI()
print("Closing's result: " + str(close_result))
