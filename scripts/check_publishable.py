#!/usr/bin/env python3
"""Pre-publish check: confirm nothing in the repo references non-public material.

This project is an independent study built on public data. The check exists so that
claim can be verified mechanically rather than trusted.

A plain grep over notebooks produces false positives, because base64-encoded PNG data
contains arbitrary byte sequences that happen to match short patterns. This script
parses notebooks and inspects only source cells and text outputs, skipping embedded
images entirely.

Usage:  python scripts/check_publishable.py
Exit code 0 means clean, 1 means something needs attention.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Terms that would indicate internal or proprietary material leaked into the repo.
PATTERNS = [
    "iqlclient", "imhotep", "presto.indeed", "skipperhive",
    "phire", "jsfb", "agg_job_id", "fccompanyid", "advertiser_id",
    "api_gt_smb", "api_events", "hiredsignalproxy", "applicationfunnel",
    "kstewart", "indeed.tech", "code.corp", "ibsci",
]
REGEX = re.compile("|".join(re.escape(p) for p in PATTERNS), re.IGNORECASE)

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pyc", ".xmind", ".zip"}


def scan_notebook(path: pathlib.Path) -> list:
    """Return (line_hint, matched_text) for hits in source cells and text outputs."""
    hits = []
    nb = json.loads(path.read_text())
    for i, cell in enumerate(nb.get("cells", [])):
        blobs = [("source", "".join(cell.get("source", [])))]
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                blobs.append(("stdout", "".join(out.get("text", []))))
            for key, val in out.get("data", {}).items():
                if key.startswith("image/"):
                    continue          # base64 payload, not human content
                blobs.append((key, "".join(val) if isinstance(val, list) else str(val)))
        for where, text in blobs:
            for m in REGEX.finditer(text):
                hits.append((f"cell {i} ({where})", m.group(0)))
    return hits


def scan_text(path: pathlib.Path) -> list:
    hits = []
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return hits
    for n, line in enumerate(text.splitlines(), 1):
        for m in REGEX.finditer(line):
            hits.append((f"line {n}", m.group(0)))
    return hits


def main() -> int:
    total = 0
    scanned = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        if path.name == pathlib.Path(__file__).name:
            continue          # this file lists the patterns by definition
        scanned += 1
        hits = scan_notebook(path) if path.suffix == ".ipynb" else scan_text(path)
        for where, match in hits:
            total += 1
            print(f"HIT  {path.relative_to(ROOT)}  {where}: {match!r}")

    print(f"\nScanned {scanned} files.")
    if total:
        print(f"{total} hit(s) need review before publishing.")
        return 1
    print("Clean: no references to non-public material.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
