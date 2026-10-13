"""Screenshot a page in headless Chrome after running some JS in it (DevTools protocol, no extra installs).

  <venv with websockets>/bin/python tools/shoot.py URL OUT.png --width 1500 --height 1100 --js 'JS' [--wait 6]

The JS runs after the page has loaded and ``--wait`` seconds have passed (so live pages fill in); it may
return a promise. The shot is the viewport at ``--width`` x ``--height`` CSS px, at 2x device pixels.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333


async def shoot(url: str, out: Path, width: int, height: int, js: str, wait: float) -> None:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/json/new?about:blank", method="PUT")
    target = json.loads(urllib.request.urlopen(req).read())
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as ws:
        seq = 0

        async def call(method: str, **params):
            nonlocal seq
            seq += 1
            await ws.send(json.dumps({"id": seq, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == seq:
                    if "error" in msg:
                        raise RuntimeError(f"{method}: {msg['error']}")
                    return msg["result"]

        await call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2, mobile=False)
        await call("Page.enable")
        await call("Page.navigate", url=url)
        await asyncio.sleep(wait)
        if js:
            res = await call("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True)
            if res.get("exceptionDetails"):
                raise RuntimeError(res["exceptionDetails"])
            if res["result"].get("value") is not None:
                print("js:", res["result"]["value"])
        shot = await call("Page.captureScreenshot", format="png")
        out.write_bytes(base64.b64decode(shot["data"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--height", type=int, default=1100)
    ap.add_argument("--js", default="")
    ap.add_argument("--wait", type=float, default=6.0)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                                   f"--user-data-dir={profile}", "--hide-scrollbars", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/version", timeout=1)
                    break
                except OSError:
                    time.sleep(0.2)
            asyncio.run(shoot(args.url, args.out, args.width, args.height, args.js, args.wait))
        finally:
            chrome.terminate()
            chrome.wait(timeout=10)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
