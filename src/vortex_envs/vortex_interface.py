"""
To interface with Vortex. 

Two main ways to exchange data with the application: its API or the dll.
The API is easier but only works w/ python 3.8. 
I dont know where the dll's doc can be found
"""
import ctypes
from pathlib import Path
from settings import APP_SETTINGS

USE_VORTEX_API = True  # To use vortex api library or the dll. Vortex api only works w/ python 3.8

if USE_VORTEX_API:
    import Vortex
    import vxatp3


class Vector3(ctypes.Structure):
    _fields_ = ('x', ctypes.c_double), ('y', ctypes.c_double), ('z', ctypes.c_double)

    def __repr__(self):
        return '({0}, {1}, {2})'.format(self.x, self.y, self.z)


class Vector4(ctypes.Structure):
    _fields_ = ('x', ctypes.c_double), ('y', ctypes.c_double), ('z', ctypes.c_double), ('w', ctypes.c_double)


class VortexInterface:
    """
    To interface with Vortex.

    Handles definitions of dll functions and their calling

    """

    def __init__(self) -> None:
        if not USE_VORTEX_API:
            self._init_vx_dll()

    def __del__(self):
        # Destroy the VxApplication when done
        self.application = None

    def _init_vx_dll(self):
        """To load the vortex dll and setup its functions types"""
        dll_path = APP_SETTINGS.vortex_installation_path / 'bin' / 'VortexIntegration.dll'

        self.vx_dll = ctypes.WinDLL(str(dll_path))

        # Declare function inputs and outputs
        self.vx_dll.VortexLoadScene.restype = ctypes.c_void_p
        self.vx_dll.VortexGetChildByName.restype = ctypes.c_void_p
        self.vx_dll.VortexGetChildByName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.vx_dll.VortexSetInputReal.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]
        self.vx_dll.VortexGetOutputReal.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_double),
        ]
        self.vx_dll.VortexGetOutputVector3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(Vector3),
        ]
        self.vx_dll.VortexGetOutputMatrix.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(Vector3),
            ctypes.POINTER(Vector4),
        ]
        self.vx_dll.VortexUnloadScene.argtypes = [ctypes.c_void_p]

    def create_application(self, setup_file: Path, application_name: str = 'Vortex App'):
        if USE_VORTEX_API:
            setup_file_str = str(setup_file)
            self.application = vxatp3.VxATPConfig.createApplication(self, application_name, setup_file_str)

            # Create a display window
            self.display = Vortex.VxExtensionFactory.create(Vortex.DisplayICD.kExtensionFactoryKey)
            self.display.getInput(Vortex.DisplayICD.kPlacementMode).setValue('Windowed')
            self.display.setName('3D Display')
            self.display.getInput(Vortex.DisplayICD.kPlacement).setValue(Vortex.VxVector4(50, 50, 1280, 720))

        else:
            self.application = self.vx_dll.VortexCreateApplication(str(setup_file).encode('ascii'), '', '', '', None)

    def load_scene(self, scene_file: Path):
        if USE_VORTEX_API:
            scene_file_str = str(scene_file)
            vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeEditing)

            self.scene = self.application.getSimulationFileManager().loadObject(scene_file_str)

            # Get the RL Interface VHL
            self.interface = self.scene.findExtensionByName('ML Interface')

            self.interface.getOutputContainer()['j2_pos_real'].value

            # Switch to Simulation Mode
            vxatp3.VxATPUtils.requestApplicationModeChangeAndWait(self.application, Vortex.kModeSimulating)

        else:
            self.scene = self.vx_dll.VortexLoadScene(str(scene_file).encode('ascii'))

        if self.scene is None or self.scene == 0:
            raise RuntimeError('Scene not properly loaded')

    def set_real_input(self, field_name, field_value):
        self.interface.getInputContainer()[field_name].value = field_value

    def get_real_output(self, field_name):
        try:
            val = self.interface.getOutputContainer()[field_name].value

        except AttributeError as err:  # If value name invalid
            raise err

        return val
