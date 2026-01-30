from argparse import Namespace

from loguru import logger

from askit.run_study.analysis import run_all_regressions
from askit.run_study.config import StudyConfig
from askit.run_study.postprocessing import postprocess_results
from askit.run_study.preprocessing import cleanup_ipc, preprocess_input


def run_study(args: Namespace) -> None:
    config = StudyConfig.from_args(args)
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
