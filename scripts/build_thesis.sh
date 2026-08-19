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

printf '\n[0a/6] Applying final terminology/metadata normalizations...\n'
# Canonical thesis notation: U+B→Y. The historical Ch2 v4 file contains one reversed-order summary occurrence.
sed -i 's/\$B+U\\rightarrow Y\$/\$U+B\\rightarrow Y\$/g' sections/02_gru_monitoring_v4.tex
# Historical placeholder text contains literal # characters; escape them even though final wrappers replace the boxes.
sed -i 's/压力#1/压力\\#1/g; s/温度#3/温度\\#3/g; s/处#1/处\\#1/g' sections/02_gru_monitoring_v4.tex
# Official NBS page displays this report as 2026-06-05.
sed -i 's/国家统计局，2026-06-04/国家统计局，2026-06-05/g' references/publication.bib

printf '\n[0b/6] Converting SVG figures to PDF...\n'
while IFS= read -r -d '' f; do
  out="${f%.svg}.pdf"
  if [[ "$SVG_CONVERTER" == "rsvg" ]]; then
    rsvg-convert -f pdf -o "$out" "$f"
  else
    inkscape "$f" --export-type=pdf --export-filename="$out" >/dev/null 2>&1
  fi
done < <(find figures -type f -name '*.svg' -print0)

# XeLaTeX/graphicx cannot reliably consume an explicitly named .svg path even when a graphics rule exists.
# Rewrite only the disposable build copy so every explicit SVG include points to the converted sibling PDF.
find sections -type f -name '*.tex' -print0 | xargs -0 sed -i 's/\.svg}/.pdf}/g'

printf '\n[1/6] XeLaTeX first pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

printf '\n[2/6] BibTeX...\n'
bibtex "$MAIN"

printf '\n[3/6] XeLaTeX second pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

printf '\n[4/6] XeLaTeX final pass...\n'
xelatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"

cp "${MAIN}.pdf" "$ROOT_DIR/${MAIN}.pdf"
cp "${MAIN}.log" "$ROOT_DIR/${MAIN}.log"

printf '\n[DONE] Generated: %s/%s.pdf\n' "$ROOT_DIR" "$MAIN"
