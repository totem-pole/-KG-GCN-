#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

MAIN="main"
BUILD_DIR="$ROOT_DIR/.thesis_build"

command -v xelatex >/dev/null 2>&1 || {
  echo "[ERROR] xelatex not found. Please install a TeX distribution with XeLaTeX support."
  exit 1
}
command -v bibtex >/dev/null 2>&1 || {
  echo "[ERROR] bibtex not found."
  exit 1
}

SVG_CONVERTER=""
if command -v rsvg-convert >/dev/null 2>&1; then
  SVG_CONVERTER="rsvg"
elif command -v inkscape >/dev/null 2>&1; then
  SVG_CONVERTER="inkscape"
else
  echo "[ERROR] neither rsvg-convert nor inkscape was found; SVG figures cannot be converted to PDF."
  exit 1
fi

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Work in a disposable copy so tracked source files are never rewritten during compilation.
cp -a main.tex sections styles references figures "$BUILD_DIR"/
cd "$BUILD_DIR"

printf '\n[0/5] Converting SVG figures to PDF...\n'
while IFS= read -r -d '' f; do
  out="${f%.svg}.pdf"
  if [[ "$SVG_CONVERTER" == "rsvg" ]]; then
    rsvg-convert -f pdf -o "$out" "$f"
  else
    inkscape "$f" --export-type=pdf --export-filename="$out" >/dev/null 2>&1
  fi
done < <(find figures -type f -name '*.svg' -print0)

printf '\n[1/5] XeLaTeX first pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

printf '\n[2/5] BibTeX...\n'
bibtex "$MAIN"

printf '\n[3/5] XeLaTeX second pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

printf '\n[4/5] XeLaTeX final pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

cp "${MAIN}.pdf" "$ROOT_DIR/${MAIN}.pdf"
cp "${MAIN}.log" "$ROOT_DIR/${MAIN}.log"

printf '\n[DONE] Generated: %s/%s.pdf\n' "$ROOT_DIR" "$MAIN"
