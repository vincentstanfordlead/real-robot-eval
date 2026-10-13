"""Build the shareable dashboard: a copy of the fleet dashboard with only what it needs.

  python3 tools/build_site.py ../inspect-robots-piper/eval_robotics_piper/fleet_logs/html/eval site

Reads the generated dashboard (``index.html`` with its data in ``<script id="data">``, plus the
``thumbs/`` and ``videos/`` it links to) and writes ``<out>/``:

- ``index.html``: the same page with the lab's name taken out of the header and the operators'
  free-text notes (``note``) dropped from every trial. The model's own notes, verdicts and
  numbers stay;
- only the thumbnails and videos the data references, copied as-is (symlinks followed).

The output is a static folder: open it with any file server. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

DATA = re.compile(r'(<script type="application/json" id="data">)(.*?)(</script>)', re.S)
BRANDING = [  # (old, new): the lab's name out of the header, and the public title in
    ("Robotics Center · Piper · YAM · xArm · eval", "Cross-embodiment LLM policy performance · Piper · YAM · xArm"),
    ("<h1>Manipulation eval</h1>", "<h1>Real-robot eval</h1>"),
    ("<title>Robot eval dashboard</title>", "<title>Real-robot eval</title>"),
]
DROP_FIELDS = ("note",)  # operator free text; everything else in a trial is generated
QUOTED_NOTE = re.compile(r' Operator: ".*"\.$', re.S)  # the form evaldata.py writes into ``reason``
MEDIA = re.compile(r"^(thumbs|videos)/")


def media_paths(obj: object) -> set[str]:
    """Every relative ``thumbs/...`` or ``videos/...`` string anywhere in the data."""
    if isinstance(obj, str):
        return {obj} if MEDIA.match(obj) else set()
    if isinstance(obj, dict):
        return set().union(*(media_paths(v) for v in obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*(media_paths(v) for v in obj)) if obj else set()
    return set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="the dashboard dir (fleet_logs/html/eval)")
    ap.add_argument("out", type=Path, help="where to write the site; replaced if it exists")
    args = ap.parse_args()

    html = (args.src / "index.html").read_text()
    m = DATA.search(html)
    if not m:
        sys.exit("no <script id=\"data\"> in index.html")
    data = json.loads(m.group(2))
    dropped = 0
    for trial in data["trials"]:
        for f in DROP_FIELDS:
            if trial.pop(f, None):
                dropped += 1
        if trial.get("reason"):  # evaldata.classify appends the note to the reason too
            trial["reason"] = QUOTED_NOTE.sub("", trial["reason"])
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = html[: m.start(2)] + blob + html[m.end(2):]
    for old, new in BRANDING:
        if old not in html:
            print(f"warning: branding string not found: {old!r}")
        html = html.replace(old, new)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    (args.out / "index.html").write_text(html)

    missing, size = [], len(html.encode())
    for rel in sorted(media_paths(data)):
        src = args.src / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)  # follows the videos -> ../../videos symlink
        size += dst.stat().st_size

    n = sum(1 for p in args.out.rglob("*") if p.is_file())
    print(f"{args.out}: {n} files, {size / 1e6:.1f} MB; {len(data['trials'])} trials, {dropped} operator notes dropped")
    if missing:
        print(f"{len(missing)} referenced files not found, e.g. {missing[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
