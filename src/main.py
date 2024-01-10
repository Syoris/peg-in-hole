from settings import APP_SETTINGS
from vortex_envs.kinova_gen2_env import KinovaGen2Env

if __name__ == '__main__':
    print('---------------- Peg-in-hole Package ----------------')
    print('Application settings:')
    print(APP_SETTINGS.model_dump())

    kinova_env = KinovaGen2Env()

    kinova_env._get_obs()
    print('Done')
