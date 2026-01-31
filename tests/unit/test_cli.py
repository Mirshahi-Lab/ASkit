from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from askit.main import main
from askit.run_study import pipeline
from askit.run_study.cli import add_run_study_command


def test_add_run_study_command_wires_func(
    sample_csv_path: Path, tmp_path: Path
) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_study_command(subparsers)
    args = parser.parse_args(
        [
            "run_study",
            "-i",
            str(sample_csv_path),
            "-o",
            str(tmp_path / "out.csv"),
            "-p",
            "pred",
            "-d",
            "dep",
            "-m",
            "linear",
        ]
    )
    assert args.func is pipeline.run_study


def test_main_requires_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["askit"])
    with pytest.raises(SystemExit):
        main()


def test_main_runs_run_study(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, object] = {}

    def fake_func(args: argparse.Namespace) -> None:
        called["called"] = True
        called["args"] = args

    def fake_parse_args(
        self: argparse.ArgumentParser, *args, **kwargs
    ) -> argparse.Namespace:
        return argparse.Namespace(func=fake_func)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    main()
    assert called.get("called") is True
