"""Seed `data/store_front.db` from the `.okf/02_store_front` bundle.

Derives a small relational schema (bundles / tools / skills / decisions /
assets) from the bundle's YAML frontmatter so the SQL role can answer
natural-language questions against a real SQLite file — no network, no
keys. Re-seeding is idempotent (DROP + CREATE).

Usage:
    python -m src.tools.seed_store_front [--db data/store_front.db]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / ".okf" / "02_store_front"
DEFAULT_DB = ROOT / "data" / "store_front.db"

SCHEMA = """
DROP TABLE IF EXISTS assets;
DROP TABLE IF EXISTS tools;
DROP TABLE IF EXISTS skills;
DROP TABLE IF EXISTS decisions;
DROP TABLE IF EXISTS bundles;

CREATE TABLE bundles (
  name        TEXT PRIMARY KEY,
  status      TEXT NOT NULL,
  description TEXT
);

CREATE TABLE tools (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  kind        TEXT NOT NULL,
  status      TEXT NOT NULL,
  path        TEXT NOT NULL,
  description TEXT,
  bundle      TEXT REFERENCES bundles(name)
);

CREATE TABLE skills (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  status      TEXT NOT NULL,
  path        TEXT NOT NULL,
  description TEXT,
  bundle      TEXT REFERENCES bundles(name)
);

CREATE TABLE decisions (
  id     TEXT PRIMARY KEY,
  title  TEXT NOT NULL,
  status TEXT NOT NULL,
  path   TEXT NOT NULL,
  summary TEXT,
  bundle TEXT REFERENCES bundles(name)
);

CREATE TABLE assets (
  bundle TEXT REFERENCES bundles(name),
  id     TEXT NOT NULL,
  type   TEXT NOT NULL,
  status TEXT NOT NULL,
  path   TEXT NOT NULL,
  PRIMARY KEY (bundle, id)
);
"""


def _frontmatter(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        return (yaml.safe_load(parts[1]) or {}), parts[2]
    except Exception:  # noqa: BLE001 — malformed frontmatter degrades to {}
        return {}, content


def _status(fm: dict) -> str:
    s = str(fm.get("status", "") or "draft")
    return s if s in ("draft", "stable", "deprecated") else "draft"


def _node_id(fm: dict, path: Path, bundle: Path) -> str:
    return str(fm.get("id") or path.relative_to(bundle).with_suffix("").as_posix())


def seed(db_path: str | Path = DEFAULT_DB, bundle_dir: str | Path = DEFAULT_BUNDLE) -> Path:
    db = Path(db_path)
    bundle = Path(bundle_dir)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)

    # ---- bundles: root .okf bundles via their index.md ----
    okf_root = bundle.parent  # .okf
    for bdir in sorted(p for p in okf_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        idx = bdir / "index.md"
        fm, _ = _frontmatter(idx) if idx.is_file() else ({}, "")
        conn.execute(
            "INSERT OR REPLACE INTO bundles (name, status, description) VALUES (?, ?, ?)",
            (bdir.name, _status(fm), str(fm.get("title", "") or "")),
        )

    # ---- 02_store_front nodes ----
    tool_rows, skill_rows, decision_rows, asset_rows = [], [], [], []
    for p in sorted(bundle.rglob("*.md")):
        fm, _ = _frontmatter(p)
        ntype = str(fm.get("type", "") or "Document")
        status = _status(fm)
        rel = p.relative_to(bundle).as_posix()
        nid = _node_id(fm, p, bundle)
        title = str(fm.get("title", "") or p.stem)
        desc = str(fm.get("description", "") or "")
        asset_rows.append((bundle.name, nid, ntype, status, rel))
        if ntype == "Tool":
            tool_rows.append((nid, title, ntype, status, rel, desc, bundle.name))
        elif ntype == "Skill":
            skill_rows.append((nid, title, status, rel, desc, bundle.name))
        elif ntype == "Decision":
            decision_rows.append((nid, title, status, rel, desc, bundle.name))

    conn.executemany(
        "INSERT INTO tools (id, name, kind, status, path, description, bundle) VALUES (?, ?, ?, ?, ?, ?, ?)",
        tool_rows,
    )
    conn.executemany(
        "INSERT INTO skills (id, name, status, path, description, bundle) VALUES (?, ?, ?, ?, ?, ?)",
        skill_rows,
    )
    conn.executemany(
        "INSERT INTO decisions (id, title, status, path, summary, bundle) VALUES (?, ?, ?, ?, ?, ?)",
        decision_rows,
    )
    conn.executemany(
        "INSERT INTO assets (bundle, id, type, status, path) VALUES (?, ?, ?, ?, ?)",
        asset_rows,
    )
    conn.commit()
    conn.close()
    return db


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    args = ap.parse_args(argv)
    db = seed(db_path=args.db, bundle_dir=args.bundle)
    print(f"seeded {db} from {args.bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
