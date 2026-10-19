"""README media from the built site: one success clip per arm, laid out like the dashboard's per-arm replay.

  <venv with cv2 + PIL>/bin/python tools/make_media.py site docs

Reads ``site/index.html`` (the dashboard data), ``site/videos/`` and ``site/thumbs/``; writes
``docs/gif/<name>.gif``, the same clip as animated WebP in ``docs/webp/`` (for the standalone HTML page) and
``docs/stats.json``. Each frame: both cameras and the Cosmos end state (the last overhead frame with the video
model's boxes and its verdict) side by side, the LLM's note at that step, the gripper opening and jaw height
with a playhead on the current step, and on the right the arm's score card.

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

# The run shown for each arm: the same task (lego brick -> box, GPT-6 Astra), a success that the operator and
# Cosmos both called, with the Cosmos end-state boxes.
ARMS = {
    "piper": {"robot": "Piper", "trial": "piper-cube-pick-place_6795303b/lego-brick-front-e0", "stride": 3, "fps": 7,
              "name": "agilex-piper"},
    "yam": {"robot": "YAM", "trial": "yam-cube-pick-place_2a9ffc96/lego-brick-front-e0", "stride": 5, "fps": 8,
            "name": "i2rt-yam"},
    "xarm": {"robot": "xArm", "trial": "xarm-cube-pick-place_307d56e8/lego-brick-front-e0", "stride": 1, "fps": 4,
             "name": "ufactory-xarm"},
}
COLOR = {"piper": "#2f6fdb", "yam": "#e8642c", "xarm": "#1f9d6b"}
PHASE = {"reach": "#2f6fdb", "close": "#e8642c", "hold": "#1f9d6b", "retreat": "#a8a29e"}
BOX = {"object": ("#22c55e", "#15803d"), "target": ("#ef4444", "#b91c1c"), "gripper": ("#38bdf8", "#0369a1")}
BG, INK, MUTED, FAINT, LINE, DARK = "#fafaf9", "#1c1917", "#78716c", "#a8a29e", "#e7e5e4", "#1c1917"
TILE_W, TILE_H = 326, 183  # one 16:9 tile; three per row (wrist, overhead, Cosmos end state)
GAP = 6
VW = 3 * TILE_W + 2 * GAP  # width of the tile row
CARD_W = 220  # score card column
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
    vlm = [r for r in judged if r.get("vlm") is not None]
    return {"trials": len(rows), "k": len(wins), "n": len(judged), "sr": len(wins) / len(judged), "ci": [lo, hi],
            "t_med": statistics.median(r["wall_s"] for r in wins if r["wall_s"] is not None),
            "arm_med": statistics.median(arm) if arm else None,
            "cosmos_agree": sum(r["vlm"] == (r["human"] == "y") for r in vlm), "cosmos_n": len(vlm)}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size * SS)


def wrap(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: float, lines: int = 2) -> list[str]:
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


def fit(img: Image.Image) -> Image.Image:
    """``img`` scaled to fit a 16:9 tile, letterboxed on black (like the dashboard's object-fit: contain)."""
    s = min(TILE_W / img.width, TILE_H / img.height)
    w, h = round(img.width * s), round(img.height * s)
    tile = Image.new("RGB", (TILE_W, TILE_H), "#000000")
    tile.paste(img.resize((w, h), Image.LANCZOS), ((TILE_W - w) // 2, (TILE_H - h) // 2))
    return tile


def read_frames(path: Path) -> list[Image.Image]:
    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return frames
        frames.append(fit(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))))


def chip(d: ImageDraw.ImageDraw, x: float, y: float, text: str, f: ImageFont.FreeTypeFont, fill: str,
         color: str = BG) -> None:
    tw = d.textlength(text, font=f)
    d.rounded_rectangle((x, y, x + tw + 10 * SS, y + 19 * SS), 3 * SS, fill=fill)
    d.text((x + 5 * SS, y + 3 * SS), text, font=f, fill=color)


def end_tile(trial: dict, site: Path) -> Image.Image:
    """The Cosmos end state: the last overhead frame with the video model's boxes (its verdict goes underneath)."""
    src = Image.open(site / trial["endframe"]["src"]).convert("RGB")
    s = min(TILE_W / src.width, TILE_H / src.height)
    w, h = round(src.width * s), round(src.height * s)
    ox, oy = (TILE_W - w) // 2, (TILE_H - h) // 2
    im = Image.new("RGB", (TILE_W * SS, TILE_H * SS), "#000000")
    im.paste(src.resize((w * SS, h * SS), Image.LANCZOS), (ox * SS, oy * SS))
    d = ImageDraw.Draw(im, "RGBA")
    lab = font(MONO, 9)
    for b in trial["endframe"]["boxes"]:
        x1, y1, x2, y2 = b["box"]
        edge, tag = BOX.get(b["role"], ("#e5e7eb", "#374151"))
        r = [(ox + x1 / 1000 * w) * SS, (oy + y1 / 1000 * h) * SS, (ox + x2 / 1000 * w) * SS, (oy + y2 / 1000 * h) * SS]
        d.rectangle(r, outline=edge, width=2 * SS)
        tw = d.textlength(b["role"], font=lab)
        ty = r[1] - 15 * SS if r[1] > 18 * SS else r[1]
        d.rectangle((r[0], ty, r[0] + tw + 8 * SS, ty + 15 * SS), fill=tag)
        d.text((r[0] + 4 * SS, ty + 2 * SS), b["role"], font=lab, fill="#ffffff")
    chip(d, 8 * SS, 8 * SS, "COSMOS END STATE", font(MONO, 10), DARK)
    return im.resize((TILE_W, TILE_H), Image.LANCZOS)


def verdict_layer(trial: dict) -> Image.Image:
    """Under the end-state tile: Cosmos's verdict next to the operator's, and its reason."""
    im = Image.new("RGB", (TILE_W * SS, CAP_H * SS), "#292524")
    d = ImageDraw.Draw(im)
    vf, rf = font(MONO, 10), font(FONT, 11)
    ok = trial["vlm"]
    head = f"COSMOS {'SUCCESS' if ok else 'FAILURE'}"
    tail = f" · operator {'success' if trial['human'] == 'y' else 'failure'} · " + \
        ("agree" if ok == (trial["human"] == "y") else "disagree")
    d.text((10 * SS, 6 * SS), head, font=vf, fill="#4ade80" if ok else "#f87171")
    d.text((10 * SS + d.textlength(head, font=vf), 6 * SS), tail, font=vf, fill="#d6d3d1")
    for j, line in enumerate(wrap(d, trial.get("vlm_why") or "", rf, (TILE_W - 20) * SS, lines=2)):
        d.text((10 * SS, (21 + j * 13) * SS), line, font=rf, fill="#e7e5e4")
    return im.resize((TILE_W, CAP_H), Image.LANCZOS)


def chart_layer(trial: dict, color: str):
    """The static chart (2 traces, plus the phase bar when it found a hold) and step -> playhead x."""
    s, segs = trial["series"], trial["segments"]
    phases = any(g["label"] == "hold" for g in segs)  # without a hold the segmentation is not meaningful
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
    lab, big, sub = font(MONO, 11), font(BOLD, 44), font(FONT, 11)
    pad, w = 20, CARD_W - 40
    y = max(46, (height - 230) // 2 + 14)
    d.line((0, 0, 0, height * SS), fill=LINE, width=SS)
    tag, tag_f = "PRELIMINARY DATA ONLY", font(MONO, 9)
    tw = d.textlength(tag, font=tag_f) / SS
    d.rounded_rectangle((pad * SS, (y - 34) * SS, (pad + tw + 14) * SS, (y - 14) * SS), 4 * SS,
                        outline="#b57a00", width=SS, fill="#fdf6e3")
    d.text(((pad + 7) * SS, (y - 30) * SS), tag, font=tag_f, fill="#8a5d00")
    d.text((pad * SS, y * SS), "SUCCESS RATE", font=lab, fill=color)
    d.text((pad * SS, (y + 18) * SS), f"{stats['sr'] * 100:.0f}%", font=big, fill=INK)
    by = y + 78
    lo, hi = stats["ci"]
    d.rounded_rectangle((pad * SS, by * SS, (pad + w) * SS, (by + 6) * SS), 3 * SS, fill=LINE)
    d.rounded_rectangle(((pad + lo * w) * SS, by * SS, (pad + hi * w) * SS, (by + 6) * SS), 3 * SS, fill=color)
    mx = pad + stats["sr"] * w
    d.rectangle(((mx - 1) * SS, (by - 4) * SS, (mx + 1) * SS, (by + 10) * SS), fill=INK)
    d.text((pad * SS, (by + 16) * SS), "human-judged · band = 95% CI", font=sub, fill=FAINT)
    y2 = by + 56
    d.text((pad * SS, y2 * SS), "TIME TO SUCCESS", font=lab, fill=color)
    d.text((pad * SS, (y2 + 18) * SS), f"{stats['t_med']:.0f} s", font=big, fill=INK)
    d.text((pad * SS, (y2 + 74) * SS), "median · every trial on this arm", font=sub, fill=FAINT)
    return im.resize((CARD_W, height), Image.LANCZOS)


def make_clip(trial: dict, stats: dict, site: Path, docs: Path, name: str, color: str, stride: int,
              fps: int) -> dict:
    views = [(v["cam"], read_frames(site / v["src"])) for v in trial["videos"]]
    n = min(len(f) for _, f in views)
    W, H = VW + CARD_W, TILE_H + CAP_H + CHART_H
    chart, head_x = chart_layer(trial, color)
    base = Image.new("RGB", (W, H), BG)
    base.paste(Image.new("RGB", (VW, TILE_H + CAP_H), DARK), (0, 0))
    base.paste(end_tile(trial, site), (2 * (TILE_W + GAP), 0))
    base.paste(verdict_layer(trial), (2 * (TILE_W + GAP), TILE_H))
    base.paste(chart, (0, TILE_H + CAP_H))
    base.paste(card_layer(stats, color, H), (VW, 0))
    label, small, cap_font = ImageFont.truetype(MONO, 11), ImageFont.truetype(MONO, 10), ImageFont.truetype(FONT, 14)
    notes = sorted(trial["notes"], key=lambda x: x["step"])
    duration = math.ceil(n / stride) / fps
    speed = trial["wall_s"] / duration
    frames = []
    for i in range(0, n, stride):
        im = base.copy()
        d = ImageDraw.Draw(im)
        d.rectangle((0, TILE_H, 2 * TILE_W + GAP - 1, TILE_H + CAP_H - 1), fill=DARK)
        for k, (cam, fr) in enumerate(views):
            x = k * (TILE_W + GAP)
            im.paste(fr[i], (x, 0))
            tw = d.textlength(cam.upper(), font=label)
            d.rounded_rectangle((x + 8, 8, x + 18 + tw, 27), 3, fill=DARK)
            d.text((x + 13, 11), cam.upper(), font=label, fill=BG)
        tag = f"{speed:.0f}× speed"
        tw, xr = d.textlength(tag, font=small), 2 * TILE_W + GAP
        d.rounded_rectangle((xr - tw - 18, 8, xr - 8, 26), 3, fill=DARK)
        d.text((xr - tw - 13, 11), tag, font=small, fill="#d6d3d1")
        note = next((x["note"] for x in reversed(notes) if x["step"] <= i and x.get("note")), "")
        for j, line in enumerate(wrap(d, note, cap_font, 2 * TILE_W + GAP - 24)):
            d.text((12, TILE_H + 6 + j * 19), line, font=cap_font, fill=BG)
        hx = head_x(i)
        d.line((hx, TILE_H + CAP_H + 6, hx, H - 6), fill=INK, width=1)
        frames.append(im)
    gif = docs / "gif" / f"{name}.gif"
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(gif, save_all=True, append_images=pal[1:], duration=round(1000 / fps), loop=0, optimize=True)
    # the same clip as animated WebP (full colour, a fraction of the size) for the standalone HTML page
    webp = docs / "webp" / f"{name}.webp"
    frames[0].save(webp, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0,
                   quality=72, method=4)
    return {"frames": len(frames), "seconds": round(duration, 1), "speed": round(speed, 1),
            "gif_bytes": gif.stat().st_size, "webp_bytes": webp.stat().st_size, "size": [W, H]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", type=Path)
    ap.add_argument("docs", type=Path)
    args = ap.parse_args()
    html = (args.site / "index.html").read_text()
    data = json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>', html, re.S).group(1))
    trials = {t["key"]: t for t in data["trials"]}
    for sub in ("gif", "webp"):
        (args.docs / sub).mkdir(parents=True, exist_ok=True)
    out = {}
    for arm, cfg in ARMS.items():
        t = trials[cfg["trial"]]
        assert t["human"] == "y" and t.get("vlm") is True and t.get("endframe"), f"{cfg['trial']}: not a clean success"
        stats = arm_stats(data["trials"], cfg["robot"])
        clip = make_clip(t, stats, args.site, args.docs, cfg["name"], COLOR[arm], cfg["stride"], cfg["fps"])
        out[arm] = {"stats": stats, "clip": clip, "trial": cfg["trial"], "embodiment": t["embodiment"],
                    "task": t["task"], "wall_s": t["wall_s"], "calls": t["calls"], "when": t["time"]}
        print(arm, json.dumps(clip), json.dumps({k: stats[k] for k in ("k", "n", "t_med")}))
    (args.docs / "stats.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
