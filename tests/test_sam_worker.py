"""
Test harness for SAM3 worker: tries different configurations to find one that
doesn't crash from double-free / heap corruption on WSL2.

Usage: uv run python tests/test_sam_worker.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TEST_IMAGE = "data/projects/Crane hook + Person/frames/video_1/video_1_000041.jpg"
CLASS_PROMPTS = ["crane hook", "person"]

JEMALLOC_PATH = Path.home() / ".local" / "lib" / "libjemalloc.so"

CONFIGS = [
    {
        "name": "jemalloc + half=False",
        "env": {"LD_PRELOAD": str(JEMALLOC_PATH)} if JEMALLOC_PATH.exists() else {},
        "half": False,
    },
    {
        "name": "jemalloc + half=True",
        "env": {"LD_PRELOAD": str(JEMALLOC_PATH)} if JEMALLOC_PATH.exists() else {},
        "half": True,
    },
    {
        "name": "tcache=0 + half=False",
        "env": {"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0"},
        "half": False,
    },
    {
        "name": "tcache=0 + half=True",
        "env": {"GLIBC_TUNABLES": "glibc.malloc.tcache_count=0"},
        "half": True,
    },
    {
        "name": "default glibc + half=False",
        "env": {},
        "half": False,
    },
    {
        "name": "default glibc + half=True",
        "env": {},
        "half": True,
    },
    {
        "name": "jemalloc + half=False + cpu",
        "env": {"LD_PRELOAD": str(JEMALLOC_PATH)} if JEMALLOC_PATH.exists() else {},
        "half": False,
        "device": "cpu",
    },
]


WORKER_TEMPLATE = '''
import json, sys
from pathlib import Path
def main():
    from ultralytics.models.sam import SAM3SemanticPredictor
    overrides = {{
        "conf": 0.25, "task": "segment", "mode": "predict",
        "model": "{model_path}", "half": {half},
        "save": False, "device": "{device}", "verbose": False,
    }}
    try:
        predictor = SAM3SemanticPredictor(overrides=overrides)
    except Exception as e:
        sys.stderr.write(f"LOAD_FAIL: {{e}}\\n")
        sys.exit(1)
    sys.stdout.write(json.dumps({{"ready": True}}) + "\\n")
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
            sys.stdout.write(json.dumps({{"detections": None, "error": str(e)}}) + "\\n")
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
                            dets.append({{"box": {{"x": float((x1+x2)/2/w), "y": float((y1+y2)/2/h), "width": float((x2-x1)/w), "height": float((y2-y1)/h)}}, "confidence": conf, "class_id": cid}})
            sys.stdout.write(json.dumps({{"detections": dets, "error": None}}) + "\\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({{"detections": None, "error": str(e)}}) + "\\n")
            sys.stdout.flush()
    return 0
if __name__ == "__main__":
    sys.exit(main())
'''


def run_trial(config: dict, project_root: Path) -> dict:
    """Run a single trial. Returns dict with result info."""
    name = config["name"]
    half = config.get("half", False)
    device = config.get("device", "auto")
    extra_env = config.get("env", {})

    model_path = project_root / "sam3.pt"
    if not model_path.exists():
        return {"name": name, "status": "SKIP", "detail": f"model not found: {model_path}"}

    script = WORKER_TEMPLATE.format(
        model_path=str(model_path),
        half=half,
        device=device,
    )
    script_path = project_root / "tests" / "_tmp_worker.py"
    script_path.write_text(script)

    env = os.environ.copy()
    env.update(extra_env)
    # Remove conflicting env vars
    if "LD_PRELOAD" in extra_env:
        env.pop("GLIBC_TUNABLES", None)
    if "GLIBC_TUNABLES" in extra_env:
        env.pop("LD_PRELOAD", None)

    print(f"\n{'='*60}")
    print(f"TRIAL: {name}")
    print(f"  half={half}, device={device}")
    print(f"  env: {extra_env}")
    print(f"{'='*60}")

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
            env=env,
        )
    except Exception as e:
        return {"name": name, "status": "FAIL", "detail": f"spawn error: {e}"}

    # Phase 1: wait for ready signal (model load)
    print("  [1/2] Waiting for model load...")
    t0 = time.time()
    try:
        proc.stdout.flush()
        import select
        ready_line = b""
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode(errors="replace")[-500:]
                elapsed = time.time() - t0
                print(f"  CRASHED during model load ({elapsed:.1f}s)")
                print(f"  stderr: {stderr}")
                return {"name": name, "status": "CRASH_LOAD", "detail": stderr.strip(), "time": elapsed}
            rlist, _, _ = select.select([proc.stdout], [], [], 1.0)
            if rlist:
                chunk = proc.stdout.readline()
                if chunk:
                    ready_line = chunk
                    break
        else:
            proc.kill()
            return {"name": name, "status": "TIMEOUT_LOAD", "detail": "model load > 120s"}
    except Exception as e:
        proc.kill()
        return {"name": name, "status": "FAIL", "detail": f"read error: {e}"}

    load_time = time.time() - t0
    try:
        msg = json.loads(ready_line.decode())
        if not msg.get("ready"):
            proc.kill()
            return {"name": name, "status": "FAIL", "detail": f"unexpected ready msg: {msg}"}
    except Exception as e:
        stderr = proc.stderr.read().decode(errors="replace")[-500:]
        proc.kill()
        return {"name": name, "status": "CRASH_LOAD", "detail": f"bad ready line: {ready_line!r} stderr: {stderr}"}

    print(f"  Model loaded in {load_time:.1f}s")

    # Phase 2: send a test inference request
    test_image = project_root / TEST_IMAGE
    if not test_image.exists():
        proc.kill()
        return {"name": name, "status": "SKIP", "detail": f"test image missing: {test_image}"}

    print(f"  [2/2] Running inference on {test_image.name}...")
    req = json.dumps({"image_path": str(test_image), "class_prompts": CLASS_PROMPTS}) + "\n"
    t1 = time.time()

    try:
        proc.stdin.write(req.encode())
        proc.stdin.flush()

        import select
        deadline = time.time() + 60
        resp_line = b""
        while time.time() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode(errors="replace")[-500:]
                elapsed = time.time() - t1
                print(f"  CRASHED during inference ({elapsed:.1f}s)")
                print(f"  stderr: {stderr}")
                return {"name": name, "status": "CRASH_INFER", "detail": stderr.strip(), "time_load": load_time, "time_infer": elapsed}
            rlist, _, _ = select.select([proc.stdout], [], [], 1.0)
            if rlist:
                chunk = proc.stdout.readline()
                if chunk:
                    resp_line = chunk
                    break
        else:
            proc.kill()
            return {"name": name, "status": "TIMEOUT_INFER", "detail": "inference > 60s", "time_load": load_time}

        infer_time = time.time() - t1
        resp = json.loads(resp_line.decode())
        proc.stdin.close()
        proc.wait(timeout=5)

        if resp.get("error"):
            print(f"  Inference returned error: {resp['error']}")
            return {"name": name, "status": "ERROR", "detail": resp["error"], "time_load": load_time, "time_infer": infer_time}

        dets = resp.get("detections", [])
        print(f"  SUCCESS: {len(dets)} detections in {infer_time:.1f}s (load: {load_time:.1f}s)")
        return {"name": name, "status": "OK", "detections": len(dets), "time_load": load_time, "time_infer": infer_time}

    except Exception as e:
        stderr = ""
        try:
            stderr = proc.stderr.read().decode(errors="replace")[-500:]
        except Exception:
            pass
        proc.kill()
        return {"name": name, "status": "FAIL", "detail": f"{e} stderr: {stderr}"}


def main():
    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")
    print(f"Test image: {project_root / TEST_IMAGE}")
    print(f"jemalloc: {JEMALLOC_PATH} (exists: {JEMALLOC_PATH.exists()})")
    print(f"Configs to test: {len(CONFIGS)}")

    results = []
    for config in CONFIGS:
        result = run_trial(config, project_root)
        results.append(result)
        # If this one worked, print it immediately
        if result["status"] == "OK":
            print(f"\n  >>> WORKING CONFIG FOUND: {result['name']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r["status"]
        name = r["name"]
        if status == "OK":
            print(f"  OK      {name} ({r['detections']} dets, load={r['time_load']:.1f}s infer={r['time_infer']:.1f}s)")
        elif "CRASH" in status:
            detail = r.get("detail", "")[:80]
            print(f"  CRASH   {name}: {detail}")
        elif "TIMEOUT" in status:
            print(f"  TIMEOUT {name}: {r.get('detail', '')}")
        else:
            detail = r.get("detail", "")[:80]
            print(f"  {status:7s} {name}: {detail}")

    winners = [r for r in results if r["status"] == "OK"]
    if winners:
        print(f"\nWINNER: {winners[0]['name']}")
    else:
        print("\nNO WORKING CONFIG FOUND")

    # Cleanup
    tmp = project_root / "tests" / "_tmp_worker.py"
    if tmp.exists():
        tmp.unlink()

    return 0 if winners else 1


if __name__ == "__main__":
    sys.exit(main())
