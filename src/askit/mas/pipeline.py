from argparse import Namespace

from loguru import logger

from askit.mas.analysis import run_all_regressions
from askit.mas.config import MASConfig
from askit.mas.postprocessing import postprocess_results
from askit.mas.preprocessing import cleanup_ipc, preprocess_input


def run_mas(args: Namespace) -> None:
    config = MASConfig.from_args(args)
    config.setup_logger()
    config.summary()
    if config.dry_run:
        return
    try:
        preprocess_input(config)
        results = run_all_regressions(config)
        postprocess_results(results, config)
    except KeyboardInterrupt:
        logger.warning("Study interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred during the study: {e}")
        raise e
    finally:
        cleanup_ipc(config.ipc_file)
