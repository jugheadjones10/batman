
import json, sys
from pathlib import Path
def main():
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = {
        "conf": 0.25, "task": "segment", "mode": "predict",
        "model": "/home/batman/batman/sam3.pt", "half": False,
        "save": False, "device": "auto", "verbose": False,
    }
    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        sys.stderr.write(f"LOAD_FAIL: {e}\n")
        sys.exit(1)
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            image_path = Path(req["image_path"])
            prompts = req.get("class_prompts", [])
        except Exception as e:
            sys.stdout.write(json.dumps({"detections": None, "error": str(e)}) + "\n")
            sys.stdout.flush()
            continue
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
            predictor.set_image(str(image_path))
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
                            dets.append({"box": {"x": float((x1+x2)/2/w), "y": float((y1+y2)/2/h), "width": float((x2-x1)/w), "height": float((y2-y1)/h)}, "confidence": conf, "class_id": cid})
            sys.stdout.write(json.dumps({"detections": dets, "error": None}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"detections": None, "error": str(e)}) + "\n")
            sys.stdout.flush()
    return 0
if __name__ == "__main__":
    sys.exit(main())
