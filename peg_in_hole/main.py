from settings import app_settings
import logging
import traceback
import hydra
from omegaconf import DictConfig

from peg_in_hole.train import train
from peg_in_hole.test import test

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name='config', config_path='../cfg')
def main(cfg: DictConfig):
    logger.info('---------------- Peg-in-hole Package ----------------')

    try:
        if cfg.run == 'train':
            train(cfg)

        elif cfg.run == 'test':
            test(cfg)

    except RuntimeError as e:
        logger.error(e, exc_info=True)
        raise e

    except KeyboardInterrupt as e:
        logger.error('KeyboardInterrupt: %s', e)

    except Exception as e:  # noqa
        logger.error('uncaught exception: %s', traceback.format_exc())
        raise e

    # finally:
    #     logger.info('Stopping neptune run')
    #     run.stop()

    logger.info('Done')


if __name__ == '__main__':
    main()
