import numpy as np
import time
from nptyping import NDArray

from ctypes import Structure, c_int, c_char, c_float, POINTER, CDLL


from robot_kg2.kinova_types.position_type import POSITION_TYPE
from robot_kg2.kinova_types.hand_command import HAND_MODE

from robot_kg2.utils import J2_CMD_LAYER_API_PATH, J2_COMM_LAYER_API_PATH
from robot_kg2.kinova_types.kinova_types import (
    KinovaDevice,
    AngularPosition,
    CartesianPosition,
    TrajectoryPoint,
)
from robot_kg2.kinova_types.error_codes import ErrorCode

import logging

logger = logging.getLogger(__name__)


class RobotKG2:
    """Class to send commands to Kinova Gen 2 robotic arm."""

    def __init__(self, verbose: bool = True) -> None:
        self.home_position: NDArray = ...

        # Robot values

        # Interface parameters
        self._kinova_dll: None | CDLL = None

    def _init_dll(self):
        logger.debug(f'Initializing kinova API dll')
        logger.debug(f'Loading dll at : {J2_CMD_LAYER_API_PATH}')
        logger.debug(f'dll functions setup')

        # Load kinova dll
        self._kinova_dll = CDLL(J2_CMD_LAYER_API_PATH)

        # Define dll function's input and output type
        self._kinova_dll.InitAPI.restype = c_int

        self._kinova_dll.CloseAPI.restype = c_int

        self._kinova_dll.GetDevices.restype = c_int
        self._kinova_dll.GetDevices.argtypes = [POINTER(KinovaDevice), POINTER(c_int)]

        self._kinova_dll.GetAngularPosition.restype = c_int
        self._kinova_dll.GetAngularPosition.argtypes = [POINTER(AngularPosition)]

        self._kinova_dll.GetAngularVelocity.restype = c_int
        self._kinova_dll.GetAngularVelocity.argtypes = [POINTER(AngularPosition)]

        self._kinova_dll.GetAngularCommand.argtypes = [POINTER(AngularPosition)]
        self._kinova_dll.GetAngularCommand.restype = c_int

        self._kinova_dll.GetAngularForce.argtypes = [POINTER(AngularPosition)]
        self._kinova_dll.GetAngularForce.restype = c_int

        self._kinova_dll.GetAngularCurrent.argtypes = [POINTER(AngularPosition)]
        self._kinova_dll.GetAngularCurrent.restype = c_int

        self._kinova_dll.GetCartesianForce.restype = c_int
        self._kinova_dll.GetCartesianForce.argtypes = [POINTER(CartesianPosition)]

        self._kinova_dll.SetActiveDevice.restype = c_int
        self._kinova_dll.SetActiveDevice.argtypes = [POINTER(KinovaDevice)]

        self._kinova_dll.SendBasicTrajectory.restype = c_int
        self._kinova_dll.SendBasicTrajectory.argtypes = [POINTER(TrajectoryPoint)]

        # Start dll
        logger.debug(f"Calling InitAPI...")
        init_res = self._kinova_dll.InitAPI()
        self._check_api_error(init_res, raise_error=False)

        logger.debug(f"Initialization's result: {ErrorCode(init_res).name}")

    def init_robot(self):
        # Init dll and connect to robot
        self._init_dll()

    # Utilities
    def _check_api_error(self, error_code, raise_error=True):
        """
        To check if a call to an api function returned an error.

        Args:
            error_code (`c_int`) : Code of the error
            raise_error (bool): Raise an error if True

        Raise:
            RuntimeError: If the error !=ErrorCode.KINOVA_NO_ERROR and `raise_error` is True
        """
        if error_code != ErrorCode.NO_ERROR_KINOVA.value:
            msg = f"API Error: {ErrorCode(error_code).name} (code: {error_code})"
            if raise_error:
                raise RuntimeError(msg)
            else:
                logger.warning(msg)

    def __del__(self):
        if self._kinova_dll is not None:
            logger.debug(f'Closing API')
            close_res = self._kinova_dll.CloseAPI()
            self._check_api_error(close_res)
