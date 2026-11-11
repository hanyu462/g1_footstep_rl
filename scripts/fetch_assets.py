"""Download the Isaac Sim G1 USD (and every layer it composes) to assets/isaac/ so runs do not stream it.

The Isaac Sim asset server is slow per connection (~0.03-0.15 MB/s), so large layers are fetched in parallel
byte ranges.  Run once in the Isaac Lab environment (pxr comes from Isaac Sim):

  python scripts/fetch_assets.py
"""
from __future__ import annotations

import argparse
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--entry", default="Robots/Unitree/G1/g1.usd", help="path below the Isaac asset root")
parser.add_argument("--connections", type=int, default=16)
args = parser.parse_args()
app = AppLauncher(headless=True).app

from pxr import Sdf  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "assets", "isaac")


def size_of(url: str) -> int:
  with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
    return int(r.headers["Content-Length"])


def fetch(url: str, dst: str, connections: int) -> None:
  n = size_of(url)
  chunk = -(-n // connections)

  def part(i: int) -> bytes:
    lo, hi = i * chunk, min(n, (i + 1) * chunk) - 1
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    for attempt in range(5):
      try:
        with urllib.request.urlopen(req, timeout=120) as r:
          return r.read()
      except OSError:
        if attempt == 4:
          raise
    raise RuntimeError
  with ThreadPoolExecutor(connections) as ex:
    data = b"".join(ex.map(part, range(-(-n // chunk))))
  assert len(data) == n, (url, len(data), n)
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  with open(dst + ".part", "wb") as f:
    f.write(data)
  os.replace(dst + ".part", dst)


def deps(path: str) -> set[str]:
  """Asset paths a layer composes (sublayers, references, payloads, also inside variants)."""
  layer = Sdf.Layer.FindOrOpen(path)
  return set() if layer is None else set(re.findall(r"@([^@]+)@", layer.ExportToString()))


def main():
  root = ISAAC_NUCLEUS_DIR
  todo, seen = [args.entry], set()
  while todo:
    rel = os.path.normpath(todo.pop())
    if rel in seen:
      continue
    seen.add(rel)
    dst = os.path.join(OUT, rel)
    if not os.path.exists(dst):
      try:
        fetch(f"{root}/{rel}", dst, args.connections)
        print(f"got  {rel}  {os.path.getsize(dst) / 1e6:.1f} MB", flush=True)
      except Exception as e:  # noqa: BLE001  (e.g. material files that are not on the server)
        print(f"miss {rel}  ({e})", flush=True)
        continue
    if dst.endswith((".usd", ".usda", ".usdc")):
      todo += [os.path.join(os.path.dirname(rel), d) for d in deps(dst) if "://" not in d]
  print(f"done: {os.path.join(OUT, args.entry)}")


if __name__ == "__main__":
  main()
  app.close()
