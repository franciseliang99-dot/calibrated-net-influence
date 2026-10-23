"""Render report/draft.md to report/report.html and, with --pdf, to report/report.pdf.

Usage: python report/build_html.py [--pdf]  (needs the `markdown` package; --pdf needs Google
Chrome, run headless with a throwaway profile). Figure paths are resolved relative to report/.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: Letter; margin: 2.2cm 2.3cm; }
body { font-family: "Times New Roman", Times, serif; font-size: 10.5pt; line-height: 1.38;
       color: #111; background: #fff; }
h1 { font-size: 16pt; text-align: center; margin: 0 0 0.4em; }
h2 { font-size: 12pt; margin: 1.1em 0 0.35em; }
h3 { font-size: 10.5pt; margin: 0.9em 0 0.3em; }
p { margin: 0.35em 0; text-align: justify; }
ul, ol { margin: 0.3em 0 0.3em 1.3em; padding: 0; }
li { margin: 0.15em 0; }
img { display: block; max-width: 100%; margin: 0.6em auto 0.2em; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 0.4em 0 0.6em; }
th, td { border-top: 0.6pt solid #888; border-bottom: 0.6pt solid #888; padding: 2pt 3pt;
         text-align: left; vertical-align: top; }
th { background: #f2f2f2; }
code { font-size: 9pt; }
h2, h3 { break-after: avoid; }
img, table { break-inside: avoid; }
"""


def main():
    md = (HERE / "draft.md").read_text()
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    title = md.splitlines()[0].lstrip("# ").strip()
    html = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{title}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    (HERE / "report.html").write_text(html)
    print(HERE / "report.html")
    if "--pdf" in sys.argv:
        with tempfile.TemporaryDirectory() as profile:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
                            f"--user-data-dir={profile}", "--no-pdf-header-footer",
                            f"--print-to-pdf={HERE / 'report.pdf'}",
                            (HERE / "report.html").as_uri()], check=True, timeout=120,
                           stderr=subprocess.DEVNULL)
        print(HERE / "report.pdf")


if __name__ == "__main__":
    main()
