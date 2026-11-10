"""Helpers for building the study guide: code excerpts pulled from the real source files,
figures from outputs/figures, and tables from the generated summary CSVs.

Nothing in the guide's code listings is retyped by hand: `src()` reads the function or class
straight from the repository, so the guide cannot drift from the code it explains.
"""

from __future__ import annotations

import ast
import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GUIDE_DIR = Path(__file__).resolve().parent


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def _find(tree: ast.AST, dotted: str) -> ast.AST:
    node = tree
    for part in dotted.split("."):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part:
                node = child
                break
        else:
            raise KeyError(f"{dotted!r} not found")
    return node


def _strip_docstring(lines: list[str], node: ast.AST, first: int) -> list[str]:
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        d0, d1 = body[0].lineno - first, body[0].end_lineno - first
        indent = lines[d0][: len(lines[d0]) - len(lines[d0].lstrip())]
        return lines[:d0] + [f'{indent}"""(docstring shortened: see the source file)"""'] + lines[d1 + 1:]
    return lines


def src(rel: str, name: str | None = None, lines: tuple[int, int] | None = None,
        short_doc: bool = False, title: str | None = None) -> str:
    """A code listing taken from `rel` (repository-relative path).

    name:  "func", "Class" or "Class.method" -> that definition (with decorators)
    lines: (first, last), 1-based inclusive
    short_doc: replace the definition's docstring with a pointer (for long docstrings)
    """
    path = ROOT / rel
    text = path.read_text()
    all_lines = text.splitlines()
    if name:
        node = _find(ast.parse(text), name)
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        last = node.end_lineno
        chunk = all_lines[first - 1:last]
        if short_doc:
            chunk = _strip_docstring(chunk, node, first)
        label = f"{rel} · {name} · lines {first}-{last}"
    elif lines:
        first, last = lines
        chunk = all_lines[first - 1:last]
        label = f"{rel} · lines {first}-{last}"
    else:
        chunk, label = all_lines, rel
    body = esc("\n".join(chunk))
    return f'<figure class="code"><figcaption>{esc(title or label)}</figcaption><pre>{body}</pre></figure>'


def pre(text: str, caption: str | None = None, cls: str = "code") -> str:
    cap = f"<figcaption>{esc(caption)}</figcaption>" if caption else ""
    return f'<figure class="{cls}">{cap}<pre>{esc(text.strip(chr(10)))}</pre></figure>'


def fig(name: str, caption: str, width: str = "100%") -> str:
    path = ROOT / "outputs" / "figures" / f"{name}.png"
    if not path.exists():
        return f'<p class="missing">[figure {esc(name)} not found: run <code>python main.py figures</code>]</p>'
    return (f'<figure class="fig"><img src="{path.as_uri()}" style="width:{width}">'
            f'<figcaption>{caption}</figcaption></figure>')


def summary(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / "results" / "summary" / f"{name}.csv")


def table(df: pd.DataFrame, caption: str | None = None, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "".join(f"<th>{esc(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, float):
                v = f"{v:.3g}"
            cells.append(f"<td>{esc(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<table>{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def box(kind: str, title: str, body: str) -> str:
    """kind: concept | why | warn | example | check"""
    return f'<div class="box {kind}"><div class="box-title">{title}</div>{body}</div>'


def qa(pairs: list[tuple[str, str]]) -> str:
    items = "".join(f'<div class="qa"><p class="q">Q. {q}</p><p class="a">{a}</p></div>' for q, a in pairs)
    return f'<div class="qas">{items}</div>'


CSS = r"""
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --ink:#1d2330; --muted:#5b6475; --accent:#1f5fa8; --accent2:#0b7a5a; --warn:#a2481b;
        --rule:#d9dee7; --code-bg:#f5f7fa; --box-bg:#f3f7fc; }
html { font-size: 10.4pt; }
body { font-family: "Charter", "Georgia", "Iowan Old Style", serif; color: var(--ink); line-height: 1.5;
       margin: 0; background: #fff; }
h1, h2, h3, h4, .box-title, figcaption, caption, th, .toc, .cover, .kicker
     { font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; }
h1 { font-size: 21pt; color: var(--accent); margin: 0 0 6pt; page-break-before: always; letter-spacing: -0.2pt; }
h1 .num { color: var(--muted); font-weight: 400; margin-right: 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 4pt; border-bottom: 1.2pt solid var(--rule); padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 13pt 0 3pt; color: #233a5e; }
h4 { font-size: 10.5pt; margin: 10pt 0 2pt; }
h2, h3, h4 { page-break-after: avoid; break-after: avoid; }
p { margin: 4pt 0 6pt; text-align: justify; hyphens: auto; }
ul, ol { margin: 3pt 0 7pt 0; padding-left: 18pt; }
li { margin: 1.5pt 0; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.6pt; background: var(--code-bg);
       padding: 0 2pt; border-radius: 2pt; }
figure { margin: 7pt 0 9pt; }
figure.code { page-break-inside: auto; border: 0.6pt solid var(--rule); border-radius: 3pt; background: var(--code-bg); }
figure.code figcaption { font-size: 7.6pt; color: var(--muted); padding: 3pt 7pt; border-bottom: 0.6pt solid var(--rule);
       background: #eef1f6; border-radius: 3pt 3pt 0 0; }
figure.code pre { margin: 0; padding: 6pt 8pt; font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 7.7pt;
       line-height: 1.36; white-space: pre-wrap; word-break: break-word; }
figure.math pre { margin: 0; padding: 6pt 10pt; font-family: "SF Mono", Menlo, monospace; font-size: 8.6pt;
       background: #fbfaf5; border-left: 2.5pt solid #c9a227; white-space: pre-wrap; }
figure.math figcaption { font-size: 8pt; color: var(--muted); margin-bottom: 2pt; }
figure.fig { text-align: center; page-break-inside: avoid; }
figure.fig img { max-width: 100%; }
figure.fig figcaption { font-size: 8.4pt; color: var(--muted); text-align: left; margin-top: 3pt; line-height: 1.35; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.6pt; width: 100%; page-break-inside: avoid; }
caption { caption-side: top; text-align: left; font-size: 8.6pt; color: var(--muted); padding-bottom: 3pt; }
th { background: #eaf0f8; text-align: left; padding: 3pt 5pt; border-bottom: 1pt solid #b8c4d6; font-weight: 600; }
td { padding: 2.5pt 5pt; border-bottom: 0.5pt solid var(--rule); vertical-align: top; }
.box { border-radius: 4pt; padding: 6pt 9pt; margin: 8pt 0 10pt; page-break-inside: avoid; border-left: 3pt solid var(--accent);
       background: var(--box-bg); }
.box p { margin: 3pt 0; }
.box-title { font-weight: 700; font-size: 9.2pt; text-transform: uppercase; letter-spacing: 0.4pt; margin-bottom: 2pt; color: var(--accent); }
.box.why { border-left-color: var(--accent2); background: #f0f8f4; } .box.why .box-title { color: var(--accent2); }
.box.warn { border-left-color: var(--warn); background: #fcf3ee; } .box.warn .box-title { color: var(--warn); }
.box.example { border-left-color: #6a4fb3; background: #f5f2fb; } .box.example .box-title { color: #6a4fb3; }
.box.check { border-left-color: #c9a227; background: #fdf9ec; } .box.check .box-title { color: #8a6d0b; }
.qa { margin: 6pt 0 8pt; page-break-inside: avoid; }
.qa .q { font-weight: 700; margin: 0; font-family: "Avenir Next", "Helvetica Neue", Arial, sans-serif; font-size: 9.6pt; }
.qa .a { margin: 1pt 0 0 12pt; }
.cover { page-break-after: always; height: 250mm; display: flex; flex-direction: column; justify-content: center; }
.cover h1 { page-break-before: avoid; font-size: 30pt; line-height: 1.12; color: var(--ink); }
.cover .sub { font-size: 14pt; color: var(--accent); margin: 8pt 0 26pt; }
.cover .meta { font-size: 9.5pt; color: var(--muted); line-height: 1.7; }
.cover .rule { height: 3pt; width: 60mm; background: var(--accent); margin: 0 0 16pt; }
.toc { page-break-after: always; }
.toc h1 { page-break-before: avoid; }
.toc ol { list-style: none; padding-left: 0; }
.toc > ol > li { margin: 6pt 0 2pt; font-weight: 600; font-size: 10.5pt; }
.toc ol ol { padding-left: 16pt; font-weight: 400; font-size: 9.3pt; columns: 2; column-gap: 18pt; }
.toc a { color: var(--ink); text-decoration: none; }
.kicker { font-size: 8.6pt; color: var(--muted); text-transform: uppercase; letter-spacing: 1pt; margin-bottom: -2pt; }
.lead { font-size: 11pt; color: #33405a; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 12pt; }
.missing { color: var(--warn); font-style: italic; }
.small { font-size: 8.8pt; color: var(--muted); }
a { color: var(--accent); }
"""


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(title)}</title>'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
