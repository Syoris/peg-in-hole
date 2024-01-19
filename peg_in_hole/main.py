from settings import app_settings
import time
import logging
import traceback
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

from peg_in_hole.ddpg.train3dof import train3dof

"""
Time comp:
For 50 steps, w/ rendering
old code: 207.0949604511261 [4.141899209022522 avg.]
new code: 113.9709882736206 [2.279419765472412 avg.]
new version: 
"""

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info('---------------- Peg-in-hole Package ----------------')
    try:
        train3dof()
    except RuntimeError as e:
        logger.error(e, exc_info=True)
        raise e

    except Exception as e:  # noqa
        logger.error('uncaught exception: %s', traceback.format_exc())
        raise e

    logger.info('Done')
