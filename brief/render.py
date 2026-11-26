"""Render brief.md to brief.pdf (2 A4 pages) with headless Chrome.

Needs the `markdown` package (not part of requirements.txt; only used here) and Google Chrome.
  python brief/render.py
"""
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 11mm 14mm 10mm 14mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.24;
       color: #111; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 1mm 0; color: #1f4e8c; }
h2 { font-size: 13pt; margin: 2.2mm 0 1.2mm 0; color: #1f4e8c; border-bottom: 1px solid #1f4e8c; }
p { margin: 0 0 1.6mm 0; }
.byline { color: #555; margin-bottom: 1.6mm; }
ul { margin: 0 0 1mm 0; padding-left: 5mm; }
li { margin-bottom: 0.9mm; }
table { border-collapse: collapse; margin: 0.6mm 0 1.6mm 0; width: 100%; }
th, td { border: 1px solid #bbb; padding: 0.35mm 2mm; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #eef4fb; }
img { width: 100%; display: block; border: 1px solid #ccc; }
img[src$=".svg"] { border: none; }
.caption { font-size: 9pt; color: #444; margin: 0.6mm 0 2mm 0; line-height: 1.2; }
.row { display: flex; gap: 4mm; align-items: flex-start; }
.col-wide { flex: 0 0 58%; }
.col-narrow { flex: 1; }
.pagebreak { break-before: page; }
.fig3 .col-wide { flex: 0 0 68%; }
code { font-size: 9.5pt; }
"""


def main():
    body = markdown.markdown((HERE / "brief.md").read_text(), extensions=["tables", "md_in_html"])
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    html_path = HERE / "brief.html"
    html_path.write_text(html)
    pdf_path = HERE / "brief.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()], check=True, capture_output=True)
    html_path.unlink()
    pages = pdf_path.read_bytes().count(b"/Type /Page") - pdf_path.read_bytes().count(b"/Type /Pages")
    print(f"{pdf_path}: {pages} pages")
    if pages != 2:
        sys.exit(1)


if __name__ == "__main__":
    main()
