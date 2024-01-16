"""
File to test the different functions and commands to control the Kinova Gen 2 robotic arm

Note that the API folder (/JACO-SDK/API/x64) must be added to the path environment in Windows
"""

from ctypes import *
import time
import numpy as np
import logging

from robot_kg2 import RobotKG2
from robot_kg2.utils import J2_CMD_LAYER_API_PATH
from robot_kg2.kinova_types import (
    KinovaDevice,
    AngularPosition,
    CartesianPosition,
    TrajectoryPoint,
)
from robot_kg2.kinova_types import ErrorCode

# Logger setup
logging.basicConfig(
    format='%(asctime)s - %(name)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%y-%b-%d %H:%M:%S'
)
logging.info(f'-------- Kinova robotic arm test --------')

MAX_KINOVA_DEVICE = 20
# SERIAL_LENGTH = 20

# Define joint starting points
j1_init = 90.0  # degrees # THIS IS 90 DEGREES DIFFERENT FROM VORTEX as we placed box at a different angle in Vortex than how robot can be clamped on table
j2_init = 184.0  # degrees
j3_init = 0.0  # degrees
j4_init = 270.0  # degrees
j5_init = 0.0  # degrees
j6_init = 180.0  # degrees
j7_init = 0.0  # degrees
j_init = np.array([j1_init, j2_init, j3_init, j4_init, j5_init, j6_init, j7_init])


# Load dll
def test_kg2():
    try:
        robot = RobotKG2()

        robot.init_robot()
        ...

    except Exception as err:
        raise err

    finally:
        logging.debug('... In finally ...')

    logging.info(f'Kinova robotic arm test completed')


if __name__ == "__main__":
    try:
        test_kg2()

    except Exception as err:
        logging.exception("Exception occurred")
