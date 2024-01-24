from dataclasses import dataclass


""" Task Config"""


@dataclass
class Vortex:
    ...


@dataclass
class Sim:
    dt: float
    vortex: Vortex


@dataclass
class Env:
    rbbot: str


@dataclass
class TaskConfigBase:
    env: Env
    sim: Sim


""" Robot Config"""


@dataclass
class Actuator:
    name: float
    position_min: float
    position_max: float
    vel_min: float
    vel_max: float
    torque_min: float
    torque_max: float


@dataclass
class KinovaConfig:
    robot_name: str
    j2: Actuator
    j3: Actuator


""" RL Config """
