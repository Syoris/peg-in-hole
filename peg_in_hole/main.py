from settings import app_settings
from peg_in_hole.vortex_envs.kinova_gen2_env import KinovaGen2Env
import time
import logging
import traceback

logger = logging.getLogger(__name__)


def train_ddpg():
    "test function"
    kinova_env = KinovaGen2Env()

    n_steps = 10

    for ep in range(n_steps):
        kinova_env.render()

        # kinova_env.reset()

        action = [5, 10, 15]

        obs = kinova_env.step(action)

        width = 10
        precision = 4
        print(f'{obs[0]:^{width}.{precision}f} | {obs[1]:^{width}.{precision}f} | {obs[2]:^{width}.{precision}f}')
        ...

    kinova_env.reset()

    del kinova_env


if __name__ == '__main__':
    logger.info('---------------- Peg-in-hole Package ----------------')
    try:
        train_ddpg()
    except RuntimeError as e:
        logger.error(e, exc_info=True)

    except:  # noqa
        logger.error('uncaught exception: %s', traceback.format_exc())

    logger.info('Done')
