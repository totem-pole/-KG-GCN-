#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MAIN="main"

command -v xelatex >/dev/null 2>&1 || {
  echo "[ERROR] xelatex not found. Please install a TeX distribution with XeLaTeX support."
  exit 1
}
command -v bibtex >/dev/null 2>&1 || {
  echo "[ERROR] bibtex not found."
  exit 1
}

printf '\n[1/4] XeLaTeX first pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

printf '\n[2/4] BibTeX...\n'
bibtex "$MAIN"

printf '\n[3/4] XeLaTeX second pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

printf '\n[4/4] XeLaTeX final pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

printf '\n[DONE] Generated: %s/%s.pdf\n' "$ROOT_DIR" "$MAIN"
