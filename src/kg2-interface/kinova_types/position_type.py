from ctypes import c_int

# Define the POSITION_TYPE type
# That represents the type of a position. If used during a trajectory, the type of position
# will change the behaviour of the robot. For example if the position type is CARTESIAN_POSITION,
# then the robot's end effector will move to that position using the inverse kinematics. But
# if the type of position is CARTESIAN_VELOCITY then the robot will use the values as velocity command.
POSITION_TYPE = c_int

# Used for initialisation.
NOMOVEMENT_POSITION = POSITION_TYPE(0)

# A cartesian position described by a translation X, Y, Z and an orientation ThetaX, thetaY and ThetaZ.
CARTESIAN_POSITION = POSITION_TYPE(1)

# An angular position described by a value for each actuator.
ANGULAR_POSITION = POSITION_TYPE(2)

# The robotic arm is in retracted mode. It may be anywhere between the HOME position and the RETRACTED position.
RETRACTED = POSITION_TYPE(3)

# The robotic arm is moving to the pre defined position #1.
PREDEFINED1 = POSITION_TYPE(4)

# The robotic arm is moving to the pre defined position #2.
PREDEFINED2 = POSITION_TYPE(5)

# The robotic arm is moving to the pre defined position #3.
PREDEFINED3 = POSITION_TYPE(6)

# A velocity vector used for velocity control.
CARTESIAN_VELOCITY = POSITION_TYPE(7)

# Used for initialisation.
ANGULAR_VELOCITY = POSITION_TYPE(8)

# The robotic arm is moving to the pre defined position #4.
PREDEFINED4 = POSITION_TYPE(9)

# The robotic arm is moving to the pre defined position #5.
PREDEFINED5 = POSITION_TYPE(10)

# Not used.
ANY_TRAJECTORY = POSITION_TYPE(11)

# The robotic arm is on time delay.
TIME_DELAY = POSITION_TYPE(12)
