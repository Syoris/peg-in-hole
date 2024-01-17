from ctypes import c_int

# Define the HAND_MODE type
# That indicates how the end effector will be used.
HAND_MODE = c_int
HAND_MODE_NOMOVEMENT = HAND_MODE(0)  # Fingers will not move.
HAND_MODE_POSITION = HAND_MODE(1)  # Fingers will move using position control.
HAND_MODE_VELOCITY = HAND_MODE(2)  # Fingers will move using velocity control.
HAND_MODE_NO_FINGER = HAND_MODE(3)
HAND_MODE_ONE_FINGER = HAND_MODE(4)
HAND_MODE_TWO_FINGERS = HAND_MODE(5)
HAND_MODE_THREE_FINGERS = HAND_MODE(6)
