"""The README as one self-contained HTML page (every image inlined), to send for review before publishing.

  python3 tools/build_page.py README.md real-robot-eval.html

Renders the small subset of Markdown the README uses (headings, paragraphs, bold / italic / links / code,
bullet lists, block images, a rule, <sub> notes; HTML comments are dropped). Animated clips use the WebP
copies in ``docs/webp/`` instead of the GIFs, and an image with an MP4 of the same name in ``docs/video/``
becomes that video (autoplay, muted, looping), which keeps the page small enough to attach to an email.
"""

from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path

MIME = {".png": "image/png", ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
        ".jpg": "image/jpeg"}

CSS = """
:root{--bg:#f6f5f1;--panel:#fff;--ink:#16161a;--sub:#55534d;--faint:#8e8c85;--line:#e2e0d9;--accent:#a4532f;
  --warn-bg:#fdf6e3;--warn:#8a5d00}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121311;--panel:#1a1b19;--ink:#efede7;
  --sub:#b0aea6;--faint:#7c7a73;--line:#2e2f2b;--accent:#e08a63;--warn-bg:#2b2410;--warn:#e0b35a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,"Segoe UI",Inter,Helvetica,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:48px 16px 72px}
h1{font-size:clamp(28px,4.4vw,44px);line-height:1.12;letter-spacing:-.02em;margin:0 0 20px}
h2{font-size:24px;letter-spacing:-.01em;margin:56px 0 12px;padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:32px 0 10px;font-family:ui-monospace,Menlo,monospace;letter-spacing:.06em;
  text-transform:uppercase;color:var(--accent)}
p{margin:0 0 14px;color:var(--sub)}
p strong,li strong{color:var(--ink)}
a{color:var(--accent)}
code{font:14px ui-monospace,Menlo,monospace;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
figure{margin:18px 0 8px}
figure img,figure video{display:block;width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#fff}
ul{margin:0 0 14px;padding-left:22px;color:var(--sub)}
li{margin:6px 0}
hr{border:0;border-top:1px solid var(--line);margin:48px 0 16px}
.small{font-size:13px;color:var(--faint)}
.small strong{color:var(--sub)}
"""


def inline(text: str) -> str:
    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<em>\1</em>", t)


def data_uri(root: Path, src: str) -> str:
    p = root / src
    if p.suffix == ".gif" and (webp := root / "docs" / "webp" / p.with_suffix(".webp").name).is_file():
        p = webp
    return f"data:{MIME[p.suffix]};base64," + base64.b64encode(p.read_bytes()).decode()


def render(md: str, root: Path) -> tuple[str, str]:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    out, title = [], ""
    for block in re.split(r"\n\s*\n", md.strip()):
        b = block.strip()
        if not b:
            continue
        if b.startswith("# "):
            title = b[2:].strip()
            out.append(f"<h1>{inline(title)}</h1>")
        elif b.startswith("## "):
            out.append(f"<h2>{inline(b[3:])}</h2>")
        elif b.startswith("### "):
            out.append(f"<h3>{inline(b[4:])}</h3>")
        elif m := re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", b):
            alt = html.escape(m.group(1))
            video = root / "docs" / "video" / (Path(m.group(2)).stem + ".mp4")
            if video.is_file():
                uri = "data:video/mp4;base64," + base64.b64encode(video.read_bytes()).decode()
                out.append(f'<figure><video src="{uri}" autoplay muted loop playsinline aria-label="{alt}"></video></figure>')
                continue
            out.append(f'<figure><img src="{data_uri(root, m.group(2))}" alt="{alt}" loading="lazy"></figure>')
        elif b == "---":
            out.append("<hr>")
        elif b.startswith("- "):
            items = re.split(r"\n- ", b[2:])
            out.append("<ul>" + "".join(f"<li>{inline(' '.join(i.split()))}</li>" for i in items) + "</ul>")
        elif b.startswith("<sub>") and b.endswith("</sub>"):
            out.append(f'<p class="small">{inline(" ".join(b[5:-6].split()))}</p>')
        else:
            out.append(f"<p>{inline(' '.join(b.split()))}</p>")
    return title, "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("readme", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    title, body = render(args.readme.read_text(), args.readme.resolve().parent)
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Real-Robot Eval</title><style>{CSS}</style></head><body>"
            f'<main>{body}</main></body></html>')
    args.out.write_text(page)
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB · {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
