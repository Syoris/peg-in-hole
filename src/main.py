from settings import APP_SETTINGS
from vortex_envs.kinova_gen2_env import KinovaGen2Env
import time


def run_vortex():
    "test function"
    kinova_env = KinovaGen2Env()

    n_steps = 100

    for ep in range(n_steps):
        kinova_env.render()

        # kinova_env.reset()

        action = [5, 10, 15]

        obs = kinova_env.step(action)

        width = 10
        precision = 4
        print(f'{obs[0]:^{width}.{precision}f} | {obs[1]:^{width}.{precision}f} | {obs[2]:^{width}.{precision}f}')
        ...

    del kinova_env


if __name__ == '__main__':
    print('---------------- Peg-in-hole Package ----------------')
    print('Application settings:')
    print(APP_SETTINGS.model_dump())

    run_vortex()

    print('Done')
