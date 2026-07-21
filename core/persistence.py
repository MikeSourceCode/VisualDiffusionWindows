"""Local state persistence via SQLite.

Stores generation presets (a snapshot of ``AppConfig``) and saved prompts so
the Streamlit app can recall previous work across sessions. The database is a
single local file; no network or user PII is stored.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Optional

from .config import AppConfig


def default_db_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "presets.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or default_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config_json TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT NOT NULL,
            negative TEXT
        )"""
    )
    conn.commit()
    return conn


def save_preset(name: str, config: AppConfig, db_path: Optional[str] = None) -> bool:
    """Save (or overwrite) a named preset. Returns False on duplicate name."""
    import json
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO presets (name, config_json) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET config_json=excluded.config_json, "
            "created_at=CURRENT_TIMESTAMP",
            (name, json.dumps(config.to_dict())),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def load_preset(name: str, db_path: Optional[str] = None) -> Optional[AppConfig]:
    import json
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT config_json FROM presets WHERE name=?", (name,)).fetchone()
        if not row:
            return None
        return AppConfig.from_dict(json.loads(row["config_json"]))
    finally:
        conn.close()


def list_presets(db_path: Optional[str] = None) -> List[Dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT name, created_at FROM presets ORDER BY created_at DESC").fetchall()
        return [{"name": r["name"], "created_at": r["created_at"]} for r in rows]
    finally:
        conn.close()


def delete_preset(name: str, db_path: Optional[str] = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM presets WHERE name=?", (name,))
        conn.commit()
    finally:
        conn.close()


def save_prompt(prompt: str, negative: str = "", label: str = "", db_path: Optional[str] = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("INSERT INTO prompts (label, prompt, negative) VALUES (?, ?, ?)",
                     (label, prompt, negative))
        conn.commit()
    finally:
        conn.close()


def list_prompts(db_path: Optional[str] = None) -> List[Dict]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, label, prompt, negative FROM prompts ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
