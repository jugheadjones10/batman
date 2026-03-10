"""Minimal SAM3 trial: load model, run one inference, exit. No imports from batman backend."""
import json
import os
import sys
from pathlib import Path

def main():
    model_path = os.environ.get("SAM_MODEL", "sam3.pt")
    device = os.environ.get("SAM_DEVICE", "auto")
    half = os.environ.get("SAM_HALF", "0") == "1"
    image_path = os.environ.get("SAM_IMAGE", "")
    prompts_str = os.environ.get("SAM_PROMPTS", "person")

    sys.stderr.write(f"trial: model={model_path} device={device} half={half} image={image_path}\n")
    sys.stderr.flush()

    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = {
        "conf": 0.25, "task": "segment", "mode": "predict",
        "model": model_path, "half": half,
        "save": False, "device": device, "verbose": False,
    }

    sys.stderr.write("trial: loading model...\n")
    sys.stderr.flush()
    predictor = SAM3SemanticPredictor(overrides=overrides)
    sys.stderr.write("trial: model loaded OK\n")
    sys.stderr.flush()

    if not image_path or not Path(image_path).exists():
        print(json.dumps({"phase": "load", "status": "ok", "detections": None}))
        return 0

    sys.stderr.write("trial: running inference...\n")
    sys.stderr.flush()
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    predictor.set_image(image_path)

    prompts = [p.strip() for p in prompts_str.split(",")]
    dets = []
    for cid, prompt in enumerate(prompts):
        results = predictor(text=[prompt])
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    xyxy = r.boxes.xyxy[i].cpu().numpy()
                    conf = float(r.boxes.conf[i].cpu().numpy()) if r.boxes.conf is not None else 1.0
                    x1, y1, x2, y2 = xyxy
                    dets.append({"class_id": cid, "confidence": conf})

    sys.stderr.write(f"trial: inference OK, {len(dets)} detections\n")
    sys.stderr.flush()
    print(json.dumps({"phase": "infer", "status": "ok", "detections": len(dets)}))
    return 0

if __name__ == "__main__":
    sys.exit(main())
