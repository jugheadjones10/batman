"""Compare the spreader's merged container box while the spreader is loaded
versus empty. Used to establish whether the load state is observable at all
from the detector's boxes. Usage: inspect_blob.py <analysis.json> <t> [t ...]
"""

import json
import sys


def main() -> None:
    frames = json.load(open(sys.argv[1]))
    for arg in sys.argv[2:]:
        t_target = float(arg)
        rec = min(frames, key=lambda x: abs(x["t"] - t_target))
        spr = next((d for d in rec["dets"] if d["cls"] == "spreader"), None)
        print(f"\nt={rec['t']:.2f}  state={rec['info']['state']}")
        if spr is None:
            print("  no spreader detection")
            continue
        b = spr["box"]
        print(
            f"  spreader#{spr['tid']:<3} cx={b['x']:.4f} cy={b['y']:.4f} "
            f"w={b['width']:.4f} h={b['height']:.4f}"
        )
        for d in rec["dets"]:
            if d["cls"] != "container":
                continue
            c = d["box"]
            print(
                f"  container#{d['tid']:<3} cx={c['x']:.4f} cy={c['y']:.4f} "
                f"w={c['width']:.4f} h={c['height']:.4f}  "
                f"dx={abs(c['x'] - b['x']):.4f} dw={abs(c['width'] - b['width']):.4f} "
                f"dh={abs(c['height'] - b['height']):.4f}"
            )


if __name__ == "__main__":
    main()
