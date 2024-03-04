import os
import sys
from enum import Enum
import math
from pathlib import Path

vortex_installation_path = os.getenv('VORTEX_PATH')
if vortex_installation_path is None:
    raise EnvironmentError(
        'Variable VORTEX_PATH not found in environment variables. Check the installation instructions.'
    )
sys.path.append(vortex_installation_path)
sys.path.append(vortex_installation_path + '/bin')

from Vortex import *  # noqa
from vxatp3 import *  # noqa

# from VxSim import *


CREATE_SCENE = True
ASSET_PATH = Path('./examples/Vortex/assets/')


class JointType(Enum):
    NONE = 0
    SPHERE = 1
    CAPSULE = 2
    CYLINDER = 3  # not available


# ___________________________________________________________________________ ________________________________________ #
# Define properties
class Socket:
    def __init__(self):
        # Dimensions
        self.thickness = 0.1  # of each plug
        self.play = 0.004  # between plug and socket
        self.tight = 0.0
        self.radius = 0.02

        self.manipulatorMass = 50
        self.plugMass = 10

        self.jointGeometry = JointType.CAPSULE

        self.xLength = 0.4  # of each connector
        self.xJointDist = 0.38  # from the base
        self.xDistance = -0.02  # to the socket

        self.yNum = 2
        self.zNum = 1

        self.ySep = 1.0
        self.zSep = 1.0

        self.yLength = self.ySep * (self.yNum - 1)
        self.zLength = self.zSep * (self.zNum - 1)

        self.yOffset = -self.yLength * 0.5
        self.zOffset = -self.zLength * 0.5

        self.zAssemblyOffset = self.zLength * 0.5 + self.zSep

        self.yBaseLength = self.yLength + self.thickness
        self.zBaseLength = self.zLength + self.thickness

        # Parts
        self.socketPart = Part.create()
        self.plugPart = Part.create()
        self.manipulatorPart = Part.create()
        self.groundPart = Part.create()

        # Joints
        self.interfaceJoint = RPRO.create()
        self.controlledJoint = RPRO.create()

        # System
        self.assembly = Assembly.create()
        self.mechanism = Mechanism.create()
        self.scene = Scene.create()

    # ________________________________________________________________________________________________________________ #
    # Create socket part with walls
    def createSocketPart(self):
        print('Creating socket part...')
        self.socketPart.setName('socket')

        for zIdx in range(self.zNum + 1):
            yPos = self.ySep * (self.yNum - 1) * 0.5 + self.yOffset
            zPos = self.zSep * (zIdx - 0.5) + self.zOffset
            ySize = self.ySep * (self.yNum) - self.thickness - self.play
            zSize = self.zSep - self.thickness - self.play

            if zIdx == 0:
                zPos = self.zSep * (zIdx - 0.25) + self.zOffset
                zSize = self.zSep * 0.5 - self.thickness - self.play

            if zIdx == self.zNum:
                zPos = self.zSep * (zIdx - 0.75) + self.zOffset
                zSize = self.zSep * 0.5 - self.thickness - self.play

            wall = Box.create()
            # wall.parameterMaterial.value = VxMaterial()
            # wall.parameterMaterial.value.setName(self.hardMaterial)
            wall.parameterDimension.value = VxVector3(self.xLength, ySize, zSize)
            wall.inputLocalTransform.value = createTranslation(VxVector3(self.xLength * 0.5, yPos, zPos))
            wall.parameterCollisionEnabled.value = True
            wall.setName('wall-z-' + str(zIdx))
            self.socketPart.addCollisionGeometry(wall)

        for yIdx in range(self.yNum + 1):
            for zIdx in range(self.zNum):
                yPos = self.ySep * (yIdx - 0.5) + self.yOffset
                zPos = self.zSep * zIdx + self.zOffset
                ySize = self.ySep - self.thickness - self.play
                zSize = self.thickness + self.play

                if yIdx == 0:
                    yPos = self.ySep * (yIdx - 0.25) + self.yOffset
                    ySize = 0.5 * self.ySep - self.thickness - self.play

                if yIdx == self.yNum:
                    yPos = self.ySep * (yIdx - 0.75) + self.yOffset
                    ySize = 0.5 * self.ySep - self.thickness - self.play

                wall = Box.create()
                # wall.parameterMaterial.value = VxMaterial()
                # wall.parameterMaterial.value.setName(self.hardMaterial)
                wall.parameterDimension.value = VxVector3(self.xLength, ySize, zSize)
                wall.inputLocalTransform.value = createTranslation(self.xLength * 0.5, yPos, zPos)
                wall.parameterCollisionEnabled.value = True
                wall.setName('wall-y-' + str(yIdx) + '-' + str(zIdx))
                self.socketPart.addCollisionGeometry(wall)

        print('Done! Socket part created. Saving part now...')

        # Save part
        save_path = ASSET_PATH / 'dynamics'
        save_path.mkdir(parents=True, exist_ok=True)

        serializer = VxObjectSerializer(self.socketPart)
        serializer.save((save_path / 'socket.vxpart').as_posix())

    # ________________________________________________________________________________________________________________ #
    # Create plug part
    def createPlugPart(self):
        print('Creating plug part...')
        self.plugPart.setName('plug')

        # Create collision geometry
        plugBase = Box.create()
        # plugBase.parameterMaterial.value = VxMaterial()
        # plugBase.parameterMaterial.value.setName(self.hardMaterial)
        plugBase.parameterDimension.value = VxVector3(self.thickness, self.yBaseLength, self.zBaseLength)
        plugBase.inputLocalTransform.value = createTranslation(
            -0.5 * self.thickness, 0.5 * self.yLength + self.yOffset, 0.5 * self.zLength + self.zOffset
        )
        plugBase.parameterCollisionEnabled.value = True
        plugBase.setName('base')
        self.plugPart.addCollisionGeometry(plugBase)

        # Create all the plugs
        for yIdx in range(self.yNum):
            for zIdx in range(self.zNum):
                yPos = yIdx * self.ySep + self.yOffset
                zPos = zIdx * self.zSep + self.zOffset

                # Create plug
                plug = Box.create()
                # plug.parameterMaterial.value = VxMaterial()
                # plug.parameterMaterial.value.setName(self.hardMaterial)
                plug.parameterDimension.value = VxVector3(self.xLength, self.thickness, self.thickness)
                plug.inputLocalTransform.value = createTranslation(self.xLength * 0.5, yPos, zPos)
                plug.parameterCollisionEnabled.value = True
                plug.setName('plug-' + str(yIdx) + '-' + str(zIdx))
                self.plugPart.addCollisionGeometry(plug)

                # Create joints
                dist = (self.thickness + self.play + self.tight) * 0.5 - self.radius

                if self.jointGeometry == JointType.NONE:
                    print('Warning: No collision geometry has been selected')

                elif self.jointGeometry == JointType.SPHERE:
                    dy = dist
                    dz = dist
                    for j in range(4):
                        joint = Sphere.create()
                        # joint.parameterMaterial.value = VxMaterial()
                        # joint.parameterMaterial.value.setName(self.softMaterial)
                        joint.parameterRadius.value = self.radius
                        joint.inputLocalTransform.value = createTranslation(self.xJointDist, yPos + dy, zPos + dz)
                        joint.parameterCollisionEnabled.value = True
                        joint.setName('plug-' + str(yIdx) + '-' + str(zIdx))
                        self.plugPart.addCollisionGeometry(joint)
                        aux = dy
                        dy = -dz
                        dz = aux

                elif self.jointGeometry == JointType.CAPSULE:
                    dy = dist
                    dz = 0
                    angle = 0.0
                    for j in range(4):
                        joint = Capsule.create()
                        # joint.parameterMaterial.value = VxMaterial()
                        # joint.parameterMaterial.value.setName(self.softMaterial)
                        joint.parameterRadius.value = self.radius
                        joint.parameterCylinderLength.value = (
                            self.thickness + self.play + self.tight - self.radius * 2.0
                        )
                        joint.inputLocalTransform.value = translateTo(
                            createRotation(angle, 0.0, 0.0), VxVector3(self.xJointDist, yPos + dy, zPos + dz)
                        )
                        if j % 2 == 0:
                            joint.parameterCollisionEnabled.value = True
                        else:
                            joint.parameterCollisionEnabled.value = False
                        joint.setName('plug-' + str(yIdx) + '-' + str(zIdx))
                        self.plugPart.addCollisionGeometry(joint)
                        aux = dy
                        dy = -dz
                        dz = aux
                        angle += math.pi * 0.5

                elif self.jointGeometry == JointType.CYLINDER:
                    print("Error: joint geometry not implemented for type 'CYLINDER'")
                    assert 0

        # Mass properties
        self.plugPart.parameterMassPropertiesContainer.mass.value = self.plugMass
        self.plugPart.autoComputeInertiaAndCOM()

        print('Done! Plug part created. Saving part now...')

        # Save part
        serializer = VxObjectSerializer(self.plugPart)

        save_path = ASSET_PATH / 'dynamics'
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = 'plug.vxpart'
        serializer.save((save_path / file_name).as_posix())

    # ________________________________________________________________________________________________________________ #
    # Create socket part with walls
    def createManipulatorPart(self):
        print('Creating manipulator part...')
        self.manipulatorPart.setName('manipulator')

        geom = Cylinder.create()
        geom.parameterHeight.value = self.xLength
        geom.parameterRadius.value = self.xLength * 0.5
        geom.inputLocalTransform.value = createRotation(0.0, math.pi * 0.5, 0.0)
        geom.parameterCollisionEnabled.value = True
        geom.setName('manipulator')
        self.manipulatorPart.addCollisionGeometry(geom)

        # Mass properties
        self.manipulatorPart.parameterMassPropertiesContainer.mass.value = self.manipulatorMass
        self.manipulatorPart.autoComputeInertiaAndCOM()

        print('Done! Manipulator part created. Saving part now...')

        # Save part
        serializer = VxObjectSerializer(self.manipulatorPart)
        save_path = ASSET_PATH / 'dynamics'
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = 'manipulator.vxpart'
        serializer.save((save_path / file_name).as_posix())

    # ________________________________________________________________________________________________________________ #
    # Create static ground part
    def createGroundPart(self):
        print('Creating ground part...')
        self.groundPart.setName('ground')

        print('Done! Ground part created. Saving part now...')

        # Save part
        serializer = VxObjectSerializer(self.manipulatorPart)
        save_path = ASSET_PATH / 'dynamics'
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = 'ground.vxpart'
        serializer.save((save_path / file_name).as_posix())

    # ________________________________________________________________________________________________________________ #
    # Create Assembly
    def createAssembly(self):
        print('Creating assembly...')
        self.assembly.setName('jamming-assembly')

        # Add Socket Part
        socketInstance = self.socketPart.instantiate()
        socketInstance.setName(self.socketPart.getName() + ' (fixed)')
        socketInstance.setLocalTransform(createTranslation(self.xLength + self.xDistance, 0.0, 0.0))
        socketInstance.inputControlType.value = Part.kControlStatic
        self.assembly.addPart(socketInstance)

        # Add Socket Part
        plugInstance = self.plugPart.instantiate()
        plugInstance.setLocalTransform(createTranslation(0.0, 0.0, 0.0))
        plugInstance.inputControlType.value = Part.kControlDynamic
        self.assembly.addPart(plugInstance)

        # Add Socket Part
        manipulatorInstance = self.manipulatorPart.instantiate()
        manipulatorInstance.setLocalTransform(createTranslation(-self.thickness - self.xLength * 0.5, 0.0, 0.0))
        manipulatorInstance.inputControlType.value = Part.kControlDynamic
        self.assembly.addPart(manipulatorInstance)

        # Add Ground Part
        groundInstance = self.groundPart.instantiate()
        groundInstance.setName(self.groundPart.getName() + ' (fixed)')
        groundInstance.setLocalTransform(createTranslation(0.0, 0.0, 0.0))
        groundInstance.inputControlType.value = Part.kControlStatic
        self.assembly.addPart(groundInstance)

        # Collision rules
        collisionRules = CollisionRuleContainerExtension.create()
        collisionRules.addRule(manipulatorInstance.getExtension(), plugInstance.getExtension())
        self.assembly.addCollisionRuleContainer(collisionRules)

        eX = VxVector3(1.0, 0.0, 0.0)
        eY = VxVector3(0.0, 1.0, 0.0)
        eZ = VxVector3(0.0, 0.0, 1.0)

        # Define interface constraints
        self.interfaceJoint.setName('interface')
        self.interfaceJoint.inputAttachment1.part.value = manipulatorInstance
        self.interfaceJoint.inputAttachment2.part.value = plugInstance
        self.interfaceJoint.inputAttachment1.position.value = VxVector3(self.xLength * 0.5, 0.0, 0.0)
        self.interfaceJoint.inputAttachment2.position.value = VxVector3(-self.thickness, 0.0, 0.0)
        self.interfaceJoint.inputAttachment1.setAxes(eX, eY)
        self.interfaceJoint.inputAttachment2.setAxes(eX, eY)
        for i in range(6):
            self.interfaceJoint.inputEquations[i].relaxation.enable.value = True
            self.interfaceJoint.inputEquations[i].relaxation.stiffness.value = VX_INFINITY
            self.interfaceJoint.inputEquations[i].relaxation.damping.value = 1e10
            self.interfaceJoint.inputEquations[i].relaxation.loss.value = 1e-10
        self.assembly.addConstraint(self.interfaceJoint)

        # Define controlled constraint
        self.controlledJoint.setName('control')
        self.controlledJoint.inputAttachment1.part.value = groundInstance
        self.controlledJoint.inputAttachment2.part.value = manipulatorInstance
        self.controlledJoint.inputAttachment1.position.value = VxVector3(-self.xLength - self.thickness, 0.0, 0.0)
        self.controlledJoint.inputAttachment2.position.value = VxVector3(-self.xLength * 0.5, 0.0, 0.0)
        self.controlledJoint.inputAttachment1.setAxes(eX, eY)
        self.controlledJoint.inputAttachment2.setAxes(eX, eY)
        for i in range(6):
            self.controlledJoint.inputEquations[i].relaxation.enable.value = True
            self.controlledJoint.inputEquations[i].relaxation.stiffness.value = VX_INFINITY
            self.controlledJoint.inputEquations[i].relaxation.damping.value = 1e10
            self.controlledJoint.inputEquations[i].relaxation.loss.value = 1e-10
        self.assembly.addConstraint(self.controlledJoint)

        print('Done! Assembly created. Saving assembly now...')

        # Save part
        serializer = VxObjectSerializer(self.assembly)
        save_path = ASSET_PATH / 'dynamics'
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = 'jamming-assem.vxassembly'
        serializer.save((save_path / file_name).as_posix())

    # ________________________________________________________________________________________________________________ #
    # Create Mechanism
    def createMechanism(self):
        print('Creating mechanism...')
        self.mechanism.setName('jamming-mechanism')

        # Add Assembly
        assemblyInstance = self.assembly.instantiate()
        assemblyInstance.setName('assembly')
        assemblyInstance.inputLocalTransform.value = createTranslation(0.0, 0.0, self.zAssemblyOffset)
        self.mechanism.addAssembly(assemblyInstance)

        print('Done! Mechanism created. Saving mechanism now...')

        # saving a definition
        serializer = VxObjectSerializer(self.mechanism)
        save_path = ASSET_PATH / 'dynamics'
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = 'jamming-mech.vxmechanism'
        serializer.save((save_path / file_name).as_posix())

    # ________________________________________________________________________________________________________________ #
    # Create Mechanism
    def createControlScript(self):
        print('Creating control script...')
        script = VxExtensionFactory.create(VxSimPythonDynamicsICD.kFactoryKey)
        script.setName('control')
        script.getParameter(VxSimPythonDynamicsICD.kScriptFile).value = './scripts/control.py'

        # Add inputs
        script.addInput('Enable', Types.Type_Bool).value = True
        script.addInput('Reset Inputs', Types.Type_Bool).value = True

        script.addInput('Linear Velocity', Types.Type_VxVector3)
        script.addInput('Angular Velocity', Types.Type_VxVector3)
        script.addInput('Max Force', Types.Type_VxVector3)
        script.addInput('Max Torque', Types.Type_VxVector3)

        # Add parameters
        script.addParameter('Reset Parameters', Types.Type_Bool).value = True

        script.addParameter('Linear Stiffness', Types.Type_VxReal).value = VX_INFINITY
        script.addParameter('Linear Damping', Types.Type_VxReal).value = 0
        script.addParameter('Angular Stiffness', Types.Type_VxReal).value = VX_INFINITY
        script.addParameter('Angular Damping', Types.Type_VxReal).value = 0

        # Add outputs
        script.addOutput('Enable', Types.Type_Bool).value = True

        script.addOutput('Linear Velocity', Types.Type_VxVector3).value = VxVector3(0, 0, 0)
        script.addOutput('Angular Velocity', Types.Type_VxVector3).value = VxVector3(0, 0, 0)

        script.addOutput('Linear Stiffness', Types.Type_VxReal).value = VX_INFINITY
        script.addOutput('Linear Damping', Types.Type_VxReal).value = 0
        script.addOutput('Linear Loss', Types.Type_VxReal).value = 0
        for i in range(3):
            script.addOutput('Max Force ' + str(i + 1), Types.Type_VxReal).value = VX_INFINITY
            script.addOutput('Min Force ' + str(i + 1), Types.Type_VxReal).value = -VX_INFINITY

        script.addOutput('Angular Stiffness', Types.Type_VxReal).value = VX_INFINITY
        script.addOutput('Angular Damping', Types.Type_VxReal).value = 0
        script.addOutput('Angular Loss', Types.Type_VxReal).value = 0
        for i in range(3):
            script.addOutput('Max Torque ' + str(i + 1), Types.Type_VxReal).value = VX_INFINITY
            script.addOutput('Min Torque ' + str(i + 1), Types.Type_VxReal).value = -VX_INFINITY

        print('Control script created.')

        return script

    # ________________________________________________________________________________________________________________ #
    # Create Mechanism
    def createControlConnectionContainer(self, controlScript):
        print('Creating connection container...')
        connectionContainer = ConnectionContainerExtension.create()
        connectionContainer.setName('connections')

        # Get current assembly and control constraint
        assemblyInstance = self.scene.getInterface().getMechanisms()[0].getAssemblies()[0]
        control = RPROInterface(assemblyInstance.getConstraints()[1].getExtension())

        connectionContainer.createConnection(
            controlScript.getOutput('Linear Velocity'), control.inputRelativeLinearVelocity
        )
        connectionContainer.createConnection(
            controlScript.getOutput('Angular Velocity'), control.inputRelativeAngularVelocity
        )

        for i in range(3):
            eqLin = control.inputEquations[i].relaxation
            eqAng = control.inputEquations[i + 3].relaxation

            connectionContainer.createConnection(controlScript.getOutput('Enable'), eqLin.enable)
            connectionContainer.createConnection(controlScript.getOutput('Linear Stiffness'), eqLin.stiffness)
            connectionContainer.createConnection(controlScript.getOutput('Linear Damping'), eqLin.damping)
            connectionContainer.createConnection(controlScript.getOutput('Linear Loss'), eqLin.loss)
            connectionContainer.createConnection(controlScript.getOutput('Max Force ' + str(i + 1)), eqLin.maximumForce)
            connectionContainer.createConnection(controlScript.getOutput('Min Force ' + str(i + 1)), eqLin.minimumForce)

            connectionContainer.createConnection(controlScript.getOutput('Enable'), eqAng.enable)
            connectionContainer.createConnection(controlScript.getOutput('Angular Stiffness'), eqAng.stiffness)
            connectionContainer.createConnection(controlScript.getOutput('Angular Damping'), eqAng.damping)
            connectionContainer.createConnection(controlScript.getOutput('Angular Loss'), eqAng.loss)
            connectionContainer.createConnection(
                controlScript.getOutput('Max Torque ' + str(i + 1)), eqAng.maximumForce
            )
            connectionContainer.createConnection(
                controlScript.getOutput('Min Torque ' + str(i + 1)), eqAng.minimumForce
            )

        print('Connection container created.')

        return connectionContainer

    # ________________________________________________________________________________________________________________ #
    # Create cosimulation extension
    def createCosimulationExtension(self):
        cosimKey = VxFactoryKey(
            VxUuid('{D2D0F2B5-B062-423C-A91E-6B51DB3FEB99}'), 'Debug', 'CoSimSplitting', 'CoSimSplitting.vxp'
        )
        cosim = VxExtensionFactory.create(cosimKey)
        cosim.setName('cosimulation')
        cosim.getInput('Interface constraint').value = self.interfaceJoint

        return cosim

    # ________________________________________________________________________________________________________________ #
    # Create data output script
    def createDataOutputScript(self):
        print('Creating data output script...')
        script = VxExtensionFactory.create(VxSimPythonDynamicsICD.kFactoryKey)
        script.setName('data-output')
        script.getParameter(VxSimPythonDynamicsICD.kScriptFile).value = './scripts/data-output.py'

        # Add inputs
        script.addInput('Enable', Types.Type_Bool).value = True

        script.addInput('control_vel_lin', Types.Type_VxVector3)
        script.addInput('control_vel_ang', Types.Type_VxVector3)
        script.addInput('max_force', Types.Type_VxVector3)
        script.addInput('max_torque', Types.Type_VxVector3)

        script.addInput('vel_lin', Types.Type_VxVector3)
        script.addInput('vel_ang', Types.Type_VxVector3)
        script.addInput('force', Types.Type_VxVector3)
        script.addInput('torque', Types.Type_VxVector3)

        script.addInput('stiffness_lin', Types.Type_VxReal)
        script.addInput('damping_lin', Types.Type_VxReal)
        script.addInput('stiffness_ang', Types.Type_VxReal)
        script.addInput('damping_ang', Types.Type_VxReal)

        print('Control script created.')

        return script

    # ________________________________________________________________________________________________________________ #
    # Create Mechanism
    def createDataOutputConnectionContainer(self, controlScript, outputScript):
        print('Creating data output connection container...')
        connectionContainer = ConnectionContainerExtension.create()
        connectionContainer.setName('connections')

        # Get current assembly and control constraint
        assemblyInstance = self.scene.getInterface().getMechanisms()[0].getAssemblies()[0]
        control = RPROInterface(assemblyInstance.getConstraints()[1].getExtension())

        connectionContainer.createConnection(
            controlScript.getOutput('Linear Velocity'), control.inputRelativeLinearVelocity
        )
        connectionContainer.createConnection(
            controlScript.getOutput('Angular Velocity'), control.inputRelativeAngularVelocity
        )

        for i in range(3):
            eqLin = control.inputEquations[i].relaxation
            eqAng = control.inputEquations[i + 3].relaxation

            connectionContainer.createConnection(controlScript.getOutput('Enable'), eqLin.enable)
            connectionContainer.createConnection(controlScript.getOutput('Linear Stiffness'), eqLin.stiffness)
            connectionContainer.createConnection(controlScript.getOutput('Linear Damping'), eqLin.damping)
            connectionContainer.createConnection(controlScript.getOutput('Linear Loss'), eqLin.loss)
            connectionContainer.createConnection(controlScript.getOutput('Max Force ' + str(i + 1)), eqLin.maximumForce)
            connectionContainer.createConnection(controlScript.getOutput('Min Force ' + str(i + 1)), eqLin.minimumForce)

            connectionContainer.createConnection(controlScript.getOutput('Enable'), eqAng.enable)
            connectionContainer.createConnection(controlScript.getOutput('Angular Stiffness'), eqAng.stiffness)
            connectionContainer.createConnection(controlScript.getOutput('Angular Damping'), eqAng.damping)
            connectionContainer.createConnection(controlScript.getOutput('Angular Loss'), eqAng.loss)
            connectionContainer.createConnection(
                controlScript.getOutput('Max Torque ' + str(i + 1)), eqAng.maximumForce
            )
            connectionContainer.createConnection(
                controlScript.getOutput('Min Torque ' + str(i + 1)), eqAng.minimumForce
            )

        print('Data output connection container created.')

        return connectionContainer

    # ________________________________________________________________________________________________________________ #
    # Create Mechanism
    def createScene(self):
        print('Creating scene...')
        self.scene.setName('jamming-scene')

        # Add Mechanism
        mechanismInstance = self.mechanism.instantiate()
        mechanismInstance.setName('mechanism')
        mechanismInstance.inputLocalTransform.value = createTranslation(0.0, 0.0, 0.0)
        self.scene.addMechanism(mechanismInstance)

        # # Add Material Table
        # key = VxMaterialTableExtensionICD.kFactoryKey
        # materialTable = VxExtensionFactory.create(key)
        # materialTable.setName('material-table')
        # materialTable.getParameter(
        #     VxMaterialTableExtensionICD.kMaterialTableFilenameParameter
        # ).value = './material-table.vxmaterials'
        # self.scene.addExtension(IMaterialTableInterface(materialTable).getExtension())

        # # Add co-simulation splitting extension
        # cosim = self.createCosimulationExtension()
        # self.scene.addExtension(cosim)

        # # Add control extension
        # controlScript = self.createControlScript()
        # self.scene.addExtension(controlScript)

        # # Create connections
        # connectionContainer = self.createControlConnectionContainer(controlScript)
        # self.scene.addExtension(connectionContainer.getExtension())

        """
        # Add control extension
        dataOutputScript = self.createDataOutputScript()
        self.scene.addExtension(dataOutputScript)

        # Create connections
        dataOutputContainer = self.createDataOutputConnectionContainer(dataOutputScript)
        self.scene.addExtension(dataOutputContainer.getExtension())
        """

        print('Done! Scene created. Saving scene now...')

        # Save scene
        serializer = VxObjectSerializer(self.scene)
        save_path = ASSET_PATH
        save_path.mkdir(parents=True, exist_ok=True)
        file_name = 'jamming-scene.vxscene'
        serializer.save((save_path / file_name).as_posix())


# ____________________________________________________________________________________________________________________ #
# ____________________________________________________________________________________________________________________ #
# MAIN SCRIPT

socket = Socket()

socket.createSocketPart()
socket.createPlugPart()
socket.createManipulatorPart()
socket.createGroundPart()

socket.createAssembly()
socket.createMechanism()

if CREATE_SCENE:
    socket.createScene()

print('===== Success! =====')
