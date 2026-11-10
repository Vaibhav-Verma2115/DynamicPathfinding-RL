"""Build the study guide PDF:  python docs/guide/build_guide.py

Assembles the chapters (content_a/b/c), adds a cover and table of contents, writes
docs/guide/guide.html and prints it to docs/DynamicPathfinding-RL_Study_Guide.pdf with headless Chrome.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import content_a  # noqa: E402
import content_b  # noqa: E402
import content_c  # noqa: E402
from guide_lib import GUIDE_DIR, ROOT, esc, page  # noqa: E402

OUT_HTML = GUIDE_DIR / "guide.html"
OUT_PDF = ROOT / "docs" / "DynamicPathfinding-RL_Study_Guide.pdf"
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "chromium", "chromium-browser",
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def add_section_ids(body: str) -> str:
    """Give every h2 an id (chN-k) so the table of contents can link to it."""
    counters: dict[str, int] = {}
    current = ["ch0"]

    def repl(m: re.Match) -> str:
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        if tag == "h1":
            cid = re.search(r'id="([^"]+)"', attrs)
            current[0] = cid.group(1) if cid else current[0]
            return m.group(0)
        counters[current[0]] = counters.get(current[0], 0) + 1
        return f'<h2 id="{current[0]}-{counters[current[0]]}"{attrs}>{inner}</h2>'

    return re.sub(r"<(h1|h2)([^>]*)>(.*?)</\1>", repl, body, flags=re.S)


def toc(body: str) -> str:
    items, open_sub = [], False
    for m in re.finditer(r'<(h1|h2) id="([^"]+)"[^>]*>(.*?)</\1>', body, flags=re.S):
        tag, hid, inner = m.groups()
        text = re.sub(r"<[^>]+>", " ", inner)
        text = re.sub(r"\s+", " ", text).strip()
        if tag == "h1":
            if open_sub:
                items.append("</ol></li>")
            items.append(f'<li><a href="#{hid}">{esc(text)}</a><ol>')
            open_sub = True
        else:
            items.append(f'<li><a href="#{hid}">{esc(text)}</a></li>')
    if open_sub:
        items.append("</ol></li>")
    return '<section class="toc"><h1>Contents</h1><ol>' + "".join(items) + "</ol></section>"


def cover() -> str:
    today = dt.date.today().strftime("%d %B %Y")
    return f"""
<section class="cover">
  <div class="kicker">Study guide and code walkthrough</div>
  <h1>DynamicPathfinding-RL</h1>
  <div class="rule"></div>
  <div class="sub">Classical pathfinding vs. model-free deep reinforcement learning in dynamic environments:
  every concept, every module, every experiment, and how to build it yourself</div>
  <div class="meta">
    Covers: grid graphs · BFS · Dijkstra · A* · D* Lite · MDPs · Value Iteration · Q-Learning · DQN · Double DQN ·
    experimental methodology · statistics · crossover analysis · Gymnasium · PyTorch · Streamlit · testing<br>
    Code listings are extracted from the repository and examples are computed by its code at build time.<br>
    Built {today} from commit <code>{git_commit()}</code>.
  </div>
</section>"""


def find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


def main() -> None:
    body = "".join(content_a.chapters() + content_b.chapters() + content_c.chapters())
    body = add_section_ids(body)
    html_text = page("DynamicPathfinding-RL Study Guide", cover() + toc(body) + body)
    OUT_HTML.write_text(html_text)
    print(f"Wrote {OUT_HTML.relative_to(ROOT)}")
    chrome = find_chrome()
    if chrome is None:
        print("Chrome/Chromium not found: open guide.html in a browser and print to PDF.")
        return
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--allow-file-access-from-files", f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
                   check=True, capture_output=True, timeout=180)
    print(f"Wrote {OUT_PDF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
