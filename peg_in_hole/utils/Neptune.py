import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)  # message=r'.*jsonschema.RefResolver is deprecated.*')

import neptune
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NeptuneRun:
    def __init__(self, neptune_cfg) -> None:
        run_name = datetime.today().strftime('%Y-%m-%d_%H-%M_PH')

        logger.info('Initialization of neptune run')
        self.run = neptune.init_run(
            project=neptune_cfg.project_name,
            api_token=neptune_cfg.api_token,
            name=run_name,
        )
        self.run.wait()
        logger.info(f"Run id: {self.run['sys/id'].fetch()}")
        logger.info(f"Run name: {self.run['sys/name'].fetch()}")

    def __del__(self):
        logger.info('Stopping neptune run')
        self.run.stop()
