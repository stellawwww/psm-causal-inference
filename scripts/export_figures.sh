#!/usr/bin/env bash
# Copy this project's figures into the portfolio site's flat images/ folder.
#
# Figures are named psm_*.png so they never collide with the images already there
# (age.jpg, fare.jpg, and so on). This script only copies files: it does not stage,
# commit, or push anything in the portfolio repo.
#
# Usage:  bash scripts/export_figures.sh [destination]

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs/figures"
DEST="${1:-$HOME/Documents/GitHub/stellaportfolio/images}"

if [ ! -d "$SRC" ]; then
  echo "No figures directory at $SRC" >&2; exit 1
fi
if [ ! -d "$DEST" ]; then
  echo "Destination not found: $DEST" >&2
  echo "Pass a different path as the first argument." >&2; exit 1
fi

count=$(find "$SRC" -maxdepth 1 -name 'psm_*.png' | wc -l | tr -d ' ')
if [ "$count" -eq 0 ]; then
  echo "No psm_*.png files to copy. Run the notebooks or scripts/run_all.py first." >&2; exit 1
fi

cp -v "$SRC"/psm_*.png "$DEST"/
echo
echo "Copied $count figure(s) to $DEST"
echo "Reference them in the portfolio page as:  <img src=\"images/psm_<name>.png\" alt=\"...\">"
