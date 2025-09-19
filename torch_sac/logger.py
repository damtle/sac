"""Lightweight CSV logger used by the PyTorch SAC trainer."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional


class CSVLogger:
    """Append-only CSV logger."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="")
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[Iterable[str]] = None

    def log(self, values: Dict[str, float]) -> None:
        if self._writer is None:
            self._fieldnames = list(values.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        assert self._fieldnames is not None
        row = {key: values.get(key) for key in self._fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()
