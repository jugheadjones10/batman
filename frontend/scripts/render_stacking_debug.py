"""Render annotated frames from the offline stacking analysis dump.

Draws every smoothed ByteTrack detection (green=matched, yellow=lost ghost)
plus the stacking-analysis roles: carried (blue), locked target (magenta),
empty-spreader nearest target (lime). Header shows state and Z readouts.

Usage: python render_stacking_debug.py <video> <analysis.json> <outdir> t1 t2 ...
"""
import json
import sys
from pathlib import Path

import cv2

video_path, analysis_path, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
times = [float(x) for x in sys.argv[4:]]
Path(outdir).mkdir(parents=True, exist_ok=True)

frames = json.load(open(analysis_path))
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

ROLE_COLORS = {
    "carried": (255, 160, 40),    # blue-ish (BGR)
    "target": (255, 60, 200),     # magenta
    "empty": (60, 230, 160),      # lime
}


def draw(frame_img, rec):
    h, w = frame_img.shape[:2]
    info = rec["info"]
    for d in rec["dets"]:
        x1 = int((d["box"]["x"] - d["box"]["width"] / 2) * w)
        y1 = int((d["box"]["y"] - d["box"]["height"] / 2) * h)
        x2 = int((d["box"]["x"] + d["box"]["width"] / 2) * w)
        y2 = int((d["box"]["y"] + d["box"]["height"] / 2) * h)
        lost = d["src"] == "lost"
        color = (0, 200, 255) if lost else (80, 220, 80)
        thick = 1 if lost else 2
        role = None
        if d["tid"] is not None:
            if d["tid"] == info["carriedTrackId"] and d["cls"] == "container":
                role, color, thick = "carried", ROLE_COLORS["carried"], 3
            elif d["tid"] == info["targetTrackId"] and d["cls"] == "container":
                role, color, thick = "target", ROLE_COLORS["target"], 3
            elif d["tid"] == info["emptyTargetTrackId"] and d["cls"] == "container":
                role, color, thick = "empty", ROLE_COLORS["empty"], 3
        cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, thick)
        z = f' z={d["z"]:.0f}' if d["z"] else ""
        label = f'{d["cls"]}#{d["tid"]}{"/lost" if lost else ""} c={d["conf"]}{z}'
        if role:
            label = f"[{role.upper()}] " + label
        cv2.putText(frame_img, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    hdr = (f't={rec["t"]:.2f}s f={rec["i"]} state={info["state"]} '
           f'carried={info["carriedTrackId"]} target={info["targetTrackId"]} '
           f'targetZ={info["targetZMm"] and round(info["targetZMm"])} '
           f'empty={info["emptyTargetTrackId"]} '
           f'emptyZ={info["emptyTargetZMm"] and round(info["emptyTargetZMm"])} '
           f'gap={info["gapMm"] and round(info["gapMm"])}')
    cv2.rectangle(frame_img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(frame_img, hdr, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return frame_img


for t in times:
    fi = int(round(t * fps))
    fi = min(fi, len(frames) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, img = cap.read()
    if not ok:
        print(f"failed to read frame {fi}")
        continue
    img = draw(img, frames[fi])
    out = f"{outdir}/t{t:07.2f}.jpg"
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out)
