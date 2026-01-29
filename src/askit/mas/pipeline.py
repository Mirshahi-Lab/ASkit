from argparse import Namespace

from askit.mas.config import MASConfig


def run_mas(args: Namespace) -> None:
    config = MASConfig.from_args(args)
    config.setup_logger()
    config.summary()
