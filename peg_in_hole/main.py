from settings import app_settings
import logging
import traceback
import hydra
from omegaconf import DictConfig

from peg_in_hole.train import train
from peg_in_hole.test import test
from peg_in_hole.utils.neptune import init_neptune_run

"""
Time comp:
For 50 steps, w/ rendering
old code: 207.0949604511261 [4.141899209022522 avg.]
new code: 113.9709882736206 [2.279419765472412 avg.]
new version: 
"""

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name='config', config_path='../cfg')
def main(cfg: DictConfig):
    logger.info('---------------- Peg-in-hole Package ----------------')

    try:
        run = init_neptune_run(cfg.run_name, neptune_cfg=cfg.neptune)

        train(cfg, run)

        test(cfg)

    except RuntimeError as e:
        logger.error(e, exc_info=True)
        raise e

    except KeyboardInterrupt as e:
        logger.error('KeyboardInterrupt: %s', e)

    except Exception as e:  # noqa
        logger.error('uncaught exception: %s', traceback.format_exc())
        raise e

    finally:
        logger.info('Stopping neptune run')
        run.stop()

    logger.info('Done')


if __name__ == '__main__':
    main()
