import argparse

from askit.mas.cli import add_mas_command


def main() -> None:
    """Entry point for the ASkit command-line interface."""
    parser = argparse.ArgumentParser(description="ASkit CLI")
    subparsers = parser.add_subparsers(
        title="subcommands", description="Available subcommands", required=True
    )
    add_mas_command(subparsers)
    # args = parser.parse_args()
