import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)  # message=r'.*jsonschema.RefResolver is deprecated.*')

import neptune  # noqa
from datetime import datetime  # noqa
import logging  # noqa

logger = logging.getLogger(__name__)


def new_neptune_run(neptune_cfg):
    run_name = datetime.today().strftime('%Y-%m-%d_%H-%M_PH')

    logger.info('Initialization of neptune run')
    run = neptune.init_run(
        project=neptune_cfg.project_name,
        api_token=neptune_cfg.api_token,
        name=run_name,
    )
    run.wait()
    logger.info(f"Run id: {run['sys/id'].fetch()}")
    logger.info(f"Run name: {run['sys/name'].fetch()}")

    return run
