"""
Package that implements the types defined in `KinovaTypes.h`

position_type.py: 
hand_commmand.py: 
3
"""

from ctypes import *
import time
import numpy as np

SERIAL_LENGTH = 20

from .position_type import POSITION_TYPE, CARTESIAN_POSITION
from .hand_command import HAND_MODE, HAND_MODE_POSITION


# Define the KinovaDevice structure
# That is a device you can communicate with via this library.
class KinovaDevice(Structure):
    _fields_ = [
        (
            "SerialNumber",
            c_char * SERIAL_LENGTH,
        ),  # The serial number of the device. If you are communicating with more than 1 device, this will be used to identify the devices.
        ("Model", c_char * SERIAL_LENGTH),  # The model of the device.
        (
            "VersionMajor",
            c_int,
        ),  # Those variables represents the code version - Major.Minor.Release
        ("VersionMinor", c_int),
        ("VersionRelease", c_int),
        ("DeviceType", c_int),  # The type of the device.
        (
            "DeviceID",
            c_int,
        ),  # This is a device ID used by the API. User should not use it.
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
        ("Actuator7", c_float),
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
    _fields_ = [("Finger1", c_float), ("Finger2", c_float), ("Finger3", c_float)]

    # This method will initialises all the values to 0:
    def InitStruct(self):
        self.Finger1 = 0.0
        self.Finger2 = 0.0
        self.Finger3 = 0.0


# Define the AngularPosition structure
# This data structure holds the values of an angular (actuators) position.
class AngularPosition(Structure):
    _fields_ = [("Actuators", AngularInfo), ("Fingers", FingersPosition)]

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
        ("X", c_float),  # Translation along X, Y, Z axes.
        ("Y", c_float),
        ("Z", c_float),
        # As an example if the current control mode is cartesian position the unit will be RAD but if the control mode is cartesian velocity
        # then the unit will be RAD per second.
        ("ThetaX", c_float),  # Orientation around X, Y, Z axes.
        ("ThetaY", c_float),
        ("ThetaZ", c_float),
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
        (
            "Coordinates",
            CartesianInfo,
        ),  # This contains values regarding the cartesian information.(end effector).
        ("Fingers", FingersPosition),  # This contains value regarding the fingers.
    ]

    # This method will initialises all the values to 0:
    def InitStruct(self):
        self.Coordinates.InitStruct()
        self.Fingers.InitStruct()


# Define the UserPosition structure, to be used in TrajectoryPoint structure
# This data structure represents an abstract position built by a user. Depending on the control type the Cartesian information, the angular information or both will be used.
class UserPosition(Structure):
    _fields_ = [
        (
            "Type",
            POSITION_TYPE,
        ),  # figure out how to implement POSITION_TYPE. The type of this position.
        (
            "Delay",
            c_float,
        ),  # This is used only if the type of position is TIME_DELAY. It represents the delay in second.
        (
            "CartesianPosition",
            CartesianInfo,
        ),  # Cartesian information about this position.
        ("Actuators", AngularInfo),  # Angular information about this position.
        (
            "HandMode",
            HAND_MODE,
        ),  # TODO: figure out how to implement HAND_MODE. Mode of the gripper.
        ("Fingers", FingersPosition),  # Fingers information about this position.
    ]

    def InitStruct(self):
        self.Type = CARTESIAN_POSITION
        self.Delay = 0.0
        self.CartesianPosition.InitStruct()
        self.Actuators.InitStruct()
        self.HandMode = HAND_MODE_POSITION
        self.Fingers.InitStruct()


# Define the Limitation structure, to be used in TrajectoryPoint structure
# This data structure represents all limitation that can be applied to a control context.
# Depending on the context, units and behaviour can change. See each parameter for more informations.
class Limitation(Structure):
    _fields_ = [
        (
            "speedParameter1",
            c_float,
        ),  # In a cartesian context, this represents the translation velocity, but in an angular context, this represents the velocity of the actuators 1, 2 and 3.
        (
            "speedParameter2",
            c_float,
        ),  # In a cartesian context, this represents the orientation velocity, but in an angular context, this represents the velocity of the actuators 4, 5 and 6.
        ("speedParameter3", c_float),  # Not used for now.
        ("forceParameter1", c_float),  # Not used for now.
        ("forceParameter2", c_float),  # Not used for now.
        ("forceParameter3", c_float),  # Not used for now.
        ("accelerationParameter1", c_float),  # Not used for now.
        ("accelerationParameter2", c_float),  # Not used for now.
        ("accelerationParameter3", c_float),  # Not used for now.
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
        (
            "Position",
            UserPosition,
        ),  # Position information that described this trajectory point.
        (
            "LimitationsActive",
            c_int,
        ),  # A flag that indicates if the limitation are active or not (1 is active 0 is not).
        (
            "SynchroType",
            c_int,
        ),  # A flag that indicates if the tracjetory's synchronization is active. (1 is active 0 is not).
        (
            "Limitations",
            Limitation,
        ),  # Limitation applied to this point if the limitation flag is active.
    ]

    def InitStruct(self):
        self.Position.InitStruct()
        self.LimitationsActive = 0
        self.SynchroType = 0
        self.Limitations.InitStruct()
