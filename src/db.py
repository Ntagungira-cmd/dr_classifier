"""SQLite persistence for uploaded retrain data."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from src.config import DB_PATH, UPLOAD_DIR


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                label INTEGER NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                used_in_retrain INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                n_images INTEGER,
                status TEXT,
                metrics_json TEXT
            )
            """
        )
        conn.commit()


def save_upload(
    filename: str,
    label: int,
    data: bytes,
    *,
    db_path: Path = DB_PATH,
) -> dict:
    init_db(db_path)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name
    dest = UPLOAD_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}_{safe}"
    dest.write_bytes(data)
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO uploads (filename, label, path, created_at) VALUES (?, ?, ?, ?)",
            (safe, int(label), str(dest), now),
        )
        conn.commit()
        row_id = cur.lastrowid
    return {"id": row_id, "filename": safe, "label": label, "path": str(dest)}


def list_uploads(unused_only: bool = False, db_path: Path = DB_PATH) -> List[sqlite3.Row]:
    init_db(db_path)
    q = "SELECT * FROM uploads"
    if unused_only:
        q += " WHERE used_in_retrain = 0"
    q += " ORDER BY id DESC"
    with _connect(db_path) as conn:
        return list(conn.execute(q))


def mark_used(ids: Iterable[int], db_path: Path = DB_PATH) -> None:
    ids = list(ids)
    if not ids:
        return
    with _connect(db_path) as conn:
        conn.executemany(
            "UPDATE uploads SET used_in_retrain = 1 WHERE id = ?",
            [(i,) for i in ids],
        )
        conn.commit()


def start_retrain_run(n_images: int, db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO retrain_runs (started_at, n_images, status) VALUES (?, ?, ?)",
            (now, n_images, "running"),
        )
        conn.commit()
        return int(cur.lastrowid)


def finish_retrain_run(
    run_id: int,
    status: str,
    metrics_json: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE retrain_runs SET finished_at=?, status=?, metrics_json=? WHERE id=?",
            (now, status, metrics_json, run_id),
        )
        conn.commit()


def upload_counts_by_label(db_path: Path = DB_PATH) -> List[Tuple[int, int]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT label, COUNT(*) AS n FROM uploads GROUP BY label ORDER BY label"
        ).fetchall()
    return [(int(r["label"]), int(r["n"])) for r in rows]
