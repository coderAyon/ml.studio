"""User-local settings and experiment metadata."""
import json
import os
from pathlib import Path
import sqlite3
from contextlib import closing


def user_directory() -> Path:
    from PySide6.QtCore import QSettings
    chosen = os.environ.get("MLSTUDIO_HOME") or QSettings("MLStudio", "MLStudio").value("storageRoot", "")
    root = Path(chosen).expanduser() if chosen else Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share")) / "MLStudio"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Store:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "history.sqlite3"
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, algorithm TEXT, result TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS final_exposures (id INTEGER PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP, fingerprint TEXT NOT NULL, split_hash TEXT NOT NULL)")

    def record_final_exposure(self, fingerprint: str, split_hash: str):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO final_exposures(fingerprint,split_hash) VALUES (?,?)", (fingerprint,split_hash))

    def final_exposure_count(self, fingerprint: str) -> int:
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute("SELECT COUNT(*) FROM final_exposures WHERE fingerprint=?", (fingerprint,)).fetchone()[0]

    def add(self, algorithm: str, result: dict):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO experiments(algorithm,result) VALUES (?,?)", (algorithm, json.dumps(result, allow_nan=False)))

    def history(self) -> list:
        with closing(sqlite3.connect(self.path)) as db, db:
            return db.execute("SELECT created,algorithm,result FROM experiments ORDER BY id DESC").fetchall()
