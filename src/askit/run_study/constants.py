from pathlib import Path

lib_path = Path(__file__).parent.parent.parent.resolve()

phecode_defs = {
    "1.2": lib_path / "resources" / "phecode_definitions1.2.parquet",
    "X": lib_path / "resources" / "phecode_definitionsX.parquet",
}
