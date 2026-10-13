"""README media from the built site: one success GIF per arm, with its action chart and score card built in.

  <venv with cv2 + PIL>/bin/python tools/make_media.py site docs

Reads ``site/index.html`` (the dashboard data) and ``site/videos/``; writes ``docs/gif/<arm>.gif`` and
``docs/stats.json``. Each GIF frame is: both cameras side by side, the model's note at that step, the gripper
opening and jaw height with a playhead on the current step, and on the right the arm's score card.

The score card uses the dashboard's own definitions (``stats()`` in the page) over every trial of that arm:
success = human verdict over the trials that ran and were judged (Wilson 95% interval drawn as a band);
time to success = median wall time of the successes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

# The run shown for each arm: the same task (lego brick -> box, GPT-6 Astra), each arm's fastest success.
ARMS = {
    "piper": {"robot": "Piper", "trial": "piper-cube-pick-place_6795303b/lego-brick-front-e0", "stride": 3, "fps": 7,
              "name": "embodiment-a"},
    "yam": {"robot": "YAM", "trial": "yam-cube-pick-place_f0c5ad93/lego-brick-front-e0", "stride": 3, "fps": 8,
            "name": "embodiment-b"},
    "xarm": {"robot": "xArm", "trial": "xarm-cube-pick-place_280afda8/lego-brick-front-e0", "stride": 1, "fps": 4,
             # the proprioceptive phases are tuned on the Piper gripper and find no hold on the xArm yet
             "phases": False, "name": "embodiment-c"},
}
COLOR = {"piper": "#2f6fdb", "yam": "#e8642c", "xarm": "#1f9d6b"}
PHASE = {"reach": "#2f6fdb", "close": "#e8642c", "hold": "#1f9d6b", "retreat": "#a8a29e"}
BG, INK, MUTED, FAINT, LINE, DARK = "#fafaf9", "#1c1917", "#78716c", "#a8a29e", "#e7e5e4", "#1c1917"
VW = 774  # width of the camera row
CARD_W = 226  # score card column
GAP = 6
CAP_H = 50
CHART_H = 128
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/Menlo.ttc"
SS = 2  # static layers are drawn at 2x and downsampled, for smooth lines and text


def wilson(k: int, n: int) -> tuple[float, float]:
    z = 1.96
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def arm_stats(trials: list[dict], robot: str) -> dict:
    rows = [r for r in trials if r["embodiment"].split(" + ")[0] == robot and r["status"] == "ok"]
    judged = [r for r in rows if r["human"] in ("y", "n")]
    wins = [r for r in judged if r["human"] == "y"]
    lo, hi = wilson(len(wins), len(judged))
    arm = [r["wall_s"] - r["llm_s"] for r in wins if r["wall_s"] is not None and r["llm_s"] is not None]
    return {"trials": len(rows), "k": len(wins), "n": len(judged), "sr": len(wins) / len(judged), "ci": [lo, hi],
            "t_med": statistics.median(r["wall_s"] for r in wins if r["wall_s"] is not None),
            "arm_med": statistics.median(arm) if arm else None}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size * SS)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int, lines: int = 2) -> list[str]:
    out, cur = [], ""
    for w in text.split():
        if draw.textlength(f"{cur} {w}".strip(), font=f) <= width:
            cur = f"{cur} {w}".strip()
            continue
        out.append(cur)
        cur = w
        if len(out) == lines:
            break
    if len(out) < lines and cur:
        out.append(cur)
    if len(out) == lines and " ".join(out) != text:
        out[-1] = out[-1].rstrip(".,") + " …"
    return out


def read_frames(path: Path, w: int) -> list[Image.Image]:
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        h = round(f.shape[0] * w / f.shape[1])
        frames.append(Image.fromarray(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)))


def chart_layer(trial: dict, color: str, phases: bool):
    """The static chart (2 traces + phase bar) and a function mapping a step to the playhead x."""
    s, segs = trial["series"], trial["segments"]
    last = max([trial.get("steps") or 1, *[g["b"] for g in segs], 1])
    L, R = 112, 10
    im = Image.new("RGB", (VW * SS, CHART_H * SS), BG)
    d = ImageDraw.Draw(im)
    lab, small = font(MONO, 10), font(FONT, 10)

    def x(st: float) -> float:
        return L + min(st, last) / last * (VW - L - R)

    for key, title, y0 in (("grip", "GRIPPER", 10), ("z", "JAW HEIGHT", 58)):
        v, ph = s[key], 40
        lo, hi = min([*v, 0] if key == "grip" else v), max(v)
        d.text((12 * SS, (y0 + 6) * SS), title, font=lab, fill=MUTED)
        d.text((12 * SS, (y0 + 22) * SS), f"{lo:.0f}–{hi:.0f} mm", font=small, fill=FAINT)
        d.line((L * SS, (y0 + ph) * SS, (VW - R) * SS, (y0 + ph) * SS), fill=LINE, width=SS)
        pts = [(x(st) * SS, (y0 + ph - (val - lo) / ((hi - lo) or 1) * (ph - 4) - 2) * SS)
               for st, val in zip(s["step"], v)]
        d.line(pts, fill=color, width=2 * SS, joint="curve")
    if phases:
        d.text((12 * SS, 106 * SS), "PHASE", font=lab, fill=MUTED)
        for g in segs:
            x0, x1 = x(g["a"] - 1) * SS, x(g["b"]) * SS
            d.rounded_rectangle((x0, 108 * SS, max(x0 + 2 * SS, x1), 116 * SS), 2 * SS, fill=PHASE[g["label"]])
    return im.resize((VW, CHART_H), Image.LANCZOS), lambda st: round(x(st))


def card_layer(stats: dict, color: str, height: int) -> Image.Image:
    """Two numbers: success rate (its Wilson interval drawn as a band) and median time to success."""
    im = Image.new("RGB", (CARD_W * SS, height * SS), BG)
    d = ImageDraw.Draw(im)
    lab, big, sub = font(MONO, 11), font(BOLD, 46), font(FONT, 11)
    pad, w = 22, CARD_W - 44
    y = max(50, (height - 236) // 2 + 16)
    d.line((0, 0, 0, height * SS), fill=LINE, width=SS)
    tag, tag_f = "PRELIMINARY DATA ONLY", font(MONO, 9)
    tw = d.textlength(tag, font=tag_f) / SS
    d.rounded_rectangle((pad * SS, (y - 34) * SS, (pad + tw + 14) * SS, (y - 14) * SS), 4 * SS,
                        outline="#b57a00", width=SS, fill="#fdf6e3")
    d.text(((pad + 7) * SS, (y - 30) * SS), tag, font=tag_f, fill="#8a5d00")
    d.text((pad * SS, y * SS), "SUCCESS RATE", font=lab, fill=color)
    d.text((pad * SS, (y + 18) * SS), f"{stats['sr'] * 100:.0f}%", font=big, fill=INK)
    by = y + 80
    lo, hi = stats["ci"]
    d.rounded_rectangle((pad * SS, by * SS, (pad + w) * SS, (by + 6) * SS), 3 * SS, fill=LINE)
    d.rounded_rectangle(((pad + lo * w) * SS, by * SS, (pad + hi * w) * SS, (by + 6) * SS), 3 * SS, fill=color)
    mx = pad + stats["sr"] * w
    d.rectangle(((mx - 1) * SS, (by - 4) * SS, (mx + 1) * SS, (by + 10) * SS), fill=INK)
    d.text((pad * SS, (by + 16) * SS), "human-judged · band = 95% CI", font=sub, fill=FAINT)
    y2 = by + 60
    d.text((pad * SS, y2 * SS), "TIME TO SUCCESS", font=lab, fill=color)
    d.text((pad * SS, (y2 + 18) * SS), f"{stats['t_med']:.0f} s", font=big, fill=INK)
    d.text((pad * SS, (y2 + 76) * SS), "median · every trial on this arm", font=sub, fill=FAINT)
    return im.resize((CARD_W, height), Image.LANCZOS)


def make_gif(trial: dict, stats: dict, site: Path, out: Path, color: str, stride: int, fps: int,
             phases: bool) -> dict:
    cam_w = (VW - GAP * (len(trial["videos"]) - 1)) // len(trial["videos"])
    views = [(v["cam"], read_frames(site / v["src"], cam_w)) for v in trial["videos"]]
    n, vh = min(len(f) for _, f in views), views[0][1][0].height
    W, H = VW + CARD_W, vh + CAP_H + CHART_H
    chart, head_x = chart_layer(trial, color, phases)
    base = Image.new("RGB", (W, H), BG)
    base.paste(chart, (0, vh + CAP_H))
    base.paste(card_layer(stats, color, H), (VW, 0))
    label, small, cap_font = ImageFont.truetype(MONO, 12), ImageFont.truetype(MONO, 11), ImageFont.truetype(FONT, 14)
    notes = sorted(trial["notes"], key=lambda x: x["step"])
    duration = math.ceil(n / stride) / fps
    speed = trial["wall_s"] / duration
    frames = []
    for i in range(0, n, stride):
        im = base.copy()
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, VW - 1, vh + CAP_H - 1), fill=DARK)
        x = 0
        for cam, fr in views:
            im.paste(fr[i], (x, 0))
            tw = d.textlength(cam.upper(), font=label)
            d.rounded_rectangle((x + 8, 8, x + 18 + tw, 28), 4, fill=DARK)
            d.text((x + 13, 11), cam.upper(), font=label, fill=BG)
            x += cam_w + GAP
        tag = f"{speed:.0f}× speed"
        tw = d.textlength(tag, font=small)
        d.rounded_rectangle((VW - tw - 20, 8, VW - 8, 27), 4, fill=DARK)
        d.text((VW - tw - 14, 11), tag, font=small, fill="#d6d3d1")
        note = next((x["note"] for x in reversed(notes) if x["step"] <= i and x.get("note")), "")
        for j, line in enumerate(wrap(d, note, cap_font, VW - 24)):
            d.text((12, vh + 6 + j * 19), line, font=cap_font, fill=BG)
        hx = head_x(i)
        d.line((hx, vh + CAP_H + 6, hx, H - 6), fill=INK, width=1)
        frames.append(im)
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return {"frames": len(frames), "seconds": round(duration, 1), "speed": round(speed, 1),
            "bytes": out.stat().st_size, "size": [W, H]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", type=Path)
    ap.add_argument("docs", type=Path)
    args = ap.parse_args()
    html = (args.site / "index.html").read_text()
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
    trials = {t["key"]: t for t in data["trials"]}
    (args.docs / "gif").mkdir(parents=True, exist_ok=True)
    out = {}
    for arm, cfg in ARMS.items():
        t = trials[cfg["trial"]]
        assert t["human"] == "y", f"{cfg['trial']} is not a human-judged success"
        stats = arm_stats(data["trials"], cfg["robot"])
        gif = make_gif(t, stats, args.site, args.docs / "gif" / f"{cfg['name']}.gif", COLOR[arm], cfg["stride"], cfg["fps"],
                       cfg.get("phases", True))
        out[arm] = {"stats": stats, "gif": gif, "trial": cfg["trial"], "embodiment": t["embodiment"],
                    "task": t["task"], "wall_s": t["wall_s"], "calls": t["calls"], "when": t["time"]}
        print(arm, json.dumps(gif))
    (args.docs / "stats.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
