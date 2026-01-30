from argparse import Namespace

from askit.mas.config import MASConfig
from askit.mas.preprocessing import preprocess_input


def run_mas(args: Namespace) -> None:
    config = MASConfig.from_args(args)
    config.setup_logger()
    config.summary()
    if config.dry_run:
        return
    preprocess_input(config)
