#!/usr/bin/env bash
# Build the paper: figures + numbers from recorded data, main.bbl, the arXiv upload
# tarball, a .docx, and a PDF preview (pandoc + headless Chrome; this machine has no TeX).
# The PDF that arXiv or Overleaf compiles from the tarball is the typeset version.
# Usage: paper/build.sh        (needs results/latest, results/replay, paper/data/sweep.json, models/gp.pkl)
set -euo pipefail
cd "$(dirname "$0")"
env -u PYTHONPATH uv run --frozen python make_figures.py >/dev/null
python3 -I make_bbl.py
mkdir -p dist build
tar -czf dist/ponics_arxiv.tar.gz main.tex main.bbl refs.bib tables/*.tex figures/*.pdf
# Preview: pandoc reads the LaTeX (macros expanded), Chrome prints it.
python3 -I - <<'PY'
import re, pathlib
tex = pathlib.Path("main.tex").read_text()
tex = re.sub(r"\\input\{(tables/results_\w+\.tex)\}", lambda m: pathlib.Path(m.group(1)).read_text(), tex)
tex = re.sub(r"figures/(\w+)\.pdf", r"figures/\1.png", tex)
pathlib.Path("build/preview.tex").write_text(tex)
PY
cp -r tables figures build/ 2>/dev/null || true
(cd build && pandoc preview.tex --citeproc --bibliography=../refs.bib --mathml -s \
   --metadata link-citations=true -c ../preview.css -o preview.html)
google-chrome --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/dist/ponics_preview.pdf" "file://$PWD/build/preview.html" 2>/dev/null
(cd build && pandoc preview.tex --citeproc --bibliography=../refs.bib -o ../dist/ponics.docx)
pdfinfo dist/ponics_preview.pdf | awk '/^Pages:/{print "preview pages:", $2}'
ls -la dist
