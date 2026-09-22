"""Simulate a load-state-aware model on an existing tracked-frames dump.

The stacking analysis treats `spreader_loaded` / `spreader_empty` classes as the
authoritative carry state (see the Z-axis guide). Retraining is the real fix,
but the consumption path can be verified now by relabelling an existing run's
spreader detections against a known ground-truth release time.

Usage:
    simulate_load_classes.py <tracked.json> <out.json> <release_seconds>

Every spreader-class detection before <release_seconds> becomes
`spreader_loaded`; every one after becomes `spreader_empty`.
"""

import json
import re
import sys


def is_spreader_class(name: str) -> bool:
    return bool(re.search(r"spreader", name, re.I)) and not bool(
        re.search(r"container", name, re.I)
    )


def main() -> None:
    tracked_path, out_path, release_s = sys.argv[1], sys.argv[2], float(sys.argv[3])
    data = json.load(open(tracked_path))
    frames = data["frames"]

    relabelled = 0
    for frame in frames:
        t = frame.get("timestamp")
        if t is None:
            continue
        for det in frame.get("detections", []):
            if not is_spreader_class(det["class_name"]):
                continue
            det["class_name"] = "spreader_loaded" if t < release_s else "spreader_empty"
            relabelled += 1

    json.dump(data, open(out_path, "w"))
    print(f"relabelled {relabelled} spreader detections, release at {release_s:.1f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
