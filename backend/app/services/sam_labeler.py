"""SAM3-based auto-labeling service using SAM3SemanticPredictor."""

import asyncio
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from backend.app.config import settings


def _sam_worker_argv():
    """Return argv for the SAM worker subprocess."""
    return [sys.executable, "-m", "backend.app.services.sam_worker"]


def _resolve_device_for_worker() -> str:
    """Resolve sam_device to an explicit value.

    Uses nvidia-smi probing instead of torch.cuda.is_available() to avoid
    initialising CUDA in the server process.  On WSL2 the DXG driver leaks
    kernel state across fork+exec, causing training subprocesses to hang if
    the parent ever touched torch.cuda.
    """
    from src.core.trainer import _probe_gpu_info

    device = settings.sam_device
    if device in ("auto", ""):
        if _probe_gpu_info() is not None:
            return "0"
        return "cpu"
    if device in ("cuda", "gpu"):
        return "0"
    return device


class SAMLabeler:
    """Auto-labeling service using SAM3SemanticPredictor for text-based segmentation."""

    def __init__(self):
        self.predictor = None
        self.device = settings.sam_device
        self.model_path = Path(settings.sam_model_path)  # e.g. ./sam3.pt in project root
        self.current_image_path: Path | None = None
        self._worker_process = None
        self._worker_lock = asyncio.Lock()

    async def load_model(self):
        """Load SAM3SemanticPredictor. Must run on main thread to avoid PyTorch double-free."""
        if self.predictor is not None:
            return

        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            logger.info(f"Loading SAM3SemanticPredictor from {self.model_path}")

            overrides = {
                "conf": 0.25,
                "task": "segment",
                "mode": "predict",
                "model": str(self.model_path),
                "half": False,
                "save": False,
                "device": self.device,
                "verbose": False,
            }

            # Load on main thread: PyTorch/CUDA can double-free when loaded in a worker thread
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            logger.info(f"SAM3SemanticPredictor loaded successfully on {self.device}")

        except ImportError as e:
            logger.error(f"SAM3SemanticPredictor not available: {e}")
            self.predictor = None
        except Exception as e:
            logger.error(f"Failed to load SAM3: {e}")
            self.predictor = None

    async def _ensure_worker(self):
        """Spawn the SAM worker subprocess if not running. Caller must hold _worker_lock."""
        if self._worker_process is not None and self._worker_process.returncode is None:
            return
        if self._worker_process is not None:
            try:
                self._worker_process.kill()
            except Exception:
                pass
            self._worker_process = None
        env = os.environ.copy()
        # Use mimalloc to replace glibc malloc, avoiding heap corruption (double-free,
        # corrupted-double-linked-list) in PyTorch/Ultralytics on WSL2.
        # Also force device=0 because LD_PRELOAD breaks torch.cuda.is_available().
        mimalloc_path = Path.home() / ".local" / "lib" / "libmimalloc.so"
        if mimalloc_path.exists():
            env["LD_PRELOAD"] = str(mimalloc_path)
            logger.info(f"SAM worker will use mimalloc: {mimalloc_path}")
        else:
            jemalloc_path = Path.home() / ".local" / "lib" / "libjemalloc.so"
            if jemalloc_path.exists():
                env["LD_PRELOAD"] = str(jemalloc_path)
                logger.info(f"SAM worker will use jemalloc: {jemalloc_path}")
            else:
                logger.warning("No alternative allocator found; SAM worker may crash on WSL2")
        # Resolve device in parent (where CUDA detection works) and pass to worker
        resolved_device = _resolve_device_for_worker()
        env["BATMAN_SAM_DEVICE"] = resolved_device
        env["BATMAN_SAM_MODEL_PATH"] = str(self.model_path)
        try:
            self._worker_process = await asyncio.create_subprocess_exec(
                *_sam_worker_argv(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd(),
                env=env,
            )
            logger.info(f"SAM worker subprocess started (mimalloc, device={resolved_device})")
        except Exception as e:
            logger.error(f"Failed to start SAM worker: {e}")
            raise RuntimeError(f"SAM worker failed to start: {e}") from e
        # Wait for worker to finish model load and signal readiness
        try:
            line = await asyncio.wait_for(self._worker_process.stdout.readline(), timeout=180.0)
            msg = json.loads(line.decode())
            if not msg.get("ready"):
                raise RuntimeError("Worker did not signal ready")
            worker_device = msg.get("device", "?")
            logger.info(f"SAM worker model loaded and ready (device={worker_device})")
        except asyncio.TimeoutError:
            stderr_out = await self._drain_worker_stderr(self._worker_process)
            logger.error(f"SAM worker timed out during model load: {stderr_out}")
            self._worker_process.kill()
            self._worker_process = None
            raise RuntimeError("SAM worker timed out loading model")
        except Exception as e:
            stderr_out = await self._drain_worker_stderr(self._worker_process)
            logger.error(f"SAM worker failed during model load: {e} | stderr: {stderr_out}")
            try:
                self._worker_process.kill()
            except Exception:
                pass
            self._worker_process = None
            raise RuntimeError(f"SAM worker crashed during model load: {e}") from e

    async def _label_frame_via_worker(
        self,
        image_path: Path,
        class_prompts: list[str],
    ) -> list[dict]:
        """Run one frame through the SAM worker subprocess. On worker crash, raises and clears worker."""
        async with self._worker_lock:
            await self._ensure_worker()
            proc = self._worker_process
            if proc.returncode is not None:
                stderr_out = await self._drain_worker_stderr(proc)
                logger.error(f"SAM worker died before request: {stderr_out}")
                self._worker_process = None
                raise RuntimeError("SAM worker exited unexpectedly; please retry auto-label")
            try:
                req = json.dumps({"image_path": str(image_path), "class_prompts": class_prompts}) + "\n"
                proc.stdin.write(req.encode())
                await proc.stdin.drain()
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=120.0)
            except (BrokenPipeError, ConnectionResetError, asyncio.TimeoutError) as e:
                stderr_out = await self._drain_worker_stderr(proc)
                logger.error(f"SAM worker crashed: {stderr_out}")
                self._worker_process = None
                raise RuntimeError("SAM worker crashed or timed out; please retry auto-label") from e
            except Exception as e:
                stderr_out = await self._drain_worker_stderr(proc)
                logger.error(f"SAM worker error ({e}): {stderr_out}")
                self._worker_process = None
                raise RuntimeError("SAM worker crashed or timed out; please retry auto-label") from e
            if not line or proc.returncode is not None:
                stderr_out = await self._drain_worker_stderr(proc)
                logger.error(f"SAM worker exited mid-request: {stderr_out}")
                self._worker_process = None
                raise RuntimeError("SAM worker exited unexpectedly; please retry auto-label")
        try:
            out = json.loads(line.decode())
        except (ValueError, json.JSONDecodeError) as e:
            self._worker_process = None
            raise RuntimeError("SAM worker crashed or timed out; please retry auto-label") from e
        if out.get("error"):
            raise RuntimeError(out["error"])
        return out.get("detections") or []

    @staticmethod
    async def _drain_worker_stderr(proc) -> str:
        """Read whatever stderr the worker wrote (non-blocking)."""
        try:
            if proc.stderr is None:
                return ""
            data = await asyncio.wait_for(proc.stderr.read(4096), timeout=1.0)
            return data.decode(errors="replace").strip()
        except Exception:
            return ""

    async def label_frame(
        self,
        image_path: Path,
        class_prompts: list[str],
        exemplars: list[dict] | None = None,
        point_prompts: list[tuple[int, int]] | None = None,
        box_prompts: list[tuple[int, int, int, int]] | None = None,
    ) -> list[dict]:
        """
        Generate labels for a single frame using SAM3.

        Args:
            image_path: Path to frame image
            class_prompts: Text descriptions of classes to detect
            exemplars: Optional exemplar annotations to guide detection
            point_prompts: Optional point prompts (x, y)
            box_prompts: Optional box prompts (x1, y1, x2, y2)

        Returns:
            List of detections with bounding boxes
        """
        if not settings.sam_in_process:
            try:
                return await self._label_frame_via_worker(image_path, class_prompts)
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(f"SAM worker failed for {image_path.name}: {e}, falling back to YOLO")
                return await self._fallback_detect(image_path, class_prompts)

        await self.load_model()

        if self.predictor is None:
            logger.warning("SAM3 not available, using fallback YOLO detection")
            return await self._fallback_detect(image_path, class_prompts)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        detections = []

        try:
            # Set the image for the predictor
            self.predictor.set_image(str(image_path))
            self.current_image_path = image_path

            # Use text prompts for each class
            for class_id, prompt in enumerate(class_prompts):
                try:
                    # Query with text prompt
                    results = self.predictor(text=[prompt])

                    if results and len(results) > 0:
                        result = results[0]

                        mask_arrays = None
                        if result.masks is not None and len(result.masks) > 0:
                            mask_arrays = result.masks.data.cpu().numpy()

                        # Extract boxes from results
                        if result.boxes is not None and len(result.boxes) > 0:
                            for i in range(len(result.boxes)):
                                xyxy = result.boxes.xyxy[i].cpu().numpy()
                                conf = (
                                    float(result.boxes.conf[i].cpu().numpy())
                                    if result.boxes.conf is not None
                                    else 1.0
                                )

                                x1, y1, x2, y2 = xyxy
                                cx = (x1 + x2) / 2 / width
                                cy = (y1 + y2) / 2 / height
                                w = (x2 - x1) / width
                                h = (y2 - y1) / height

                                det = {
                                    "box": {
                                        "x": float(cx),
                                        "y": float(cy),
                                        "width": float(w),
                                        "height": float(h),
                                    },
                                    "confidence": conf,
                                    "class_id": class_id,
                                }
                                if mask_arrays is not None and i < len(mask_arrays):
                                    poly = self._mask_to_polygon_norm(
                                        mask_arrays[i], width, height
                                    )
                                    if poly is not None:
                                        det["polygon"] = poly
                                detections.append(det)

                        elif mask_arrays is not None:
                            for mask in mask_arrays:
                                bbox = self._mask_to_bbox(mask, width, height)
                                if bbox:
                                    det = {
                                        "box": bbox,
                                        "confidence": 1.0,
                                        "class_id": class_id,
                                    }
                                    poly = self._mask_to_polygon_norm(mask, width, height)
                                    if poly is not None:
                                        det["polygon"] = poly
                                    detections.append(det)

                except Exception as e:
                    logger.warning(f"SAM3 failed for prompt '{prompt}': {e}")
                    continue

            logger.info(f"SAM3 detected {len(detections)} objects in {image_path.name}")
            return detections

        except Exception as e:
            logger.error(f"SAM3 inference failed: {e}")
            return await self._fallback_detect(image_path, class_prompts)

    async def label_frame_with_points(
        self,
        image_path: Path,
        points: list[tuple[int, int]],
        labels: list[int],  # 1 for foreground, 0 for background
        class_id: int = 0,
    ) -> list[dict]:
        """
        Label frame using point prompts.

        Args:
            image_path: Path to frame image
            points: List of (x, y) point coordinates
            labels: List of labels (1=foreground, 0=background)
            class_id: Class ID to assign to detections

        Returns:
            List of detections
        """
        await self.load_model()

        if self.predictor is None:
            return []

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        try:
            self.predictor.set_image(str(image_path))

            # SAM3 point prompt format
            results = self.predictor(points=points, labels=labels)

            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    for mask in masks:
                        bbox = self._mask_to_bbox(mask, width, height)
                        if bbox:
                            det = {
                                "box": bbox,
                                "confidence": 1.0,
                                "class_id": class_id,
                            }
                            poly = self._mask_to_polygon_norm(mask, width, height)
                            if poly is not None:
                                det["polygon"] = poly
                            detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"SAM3 point prompt failed: {e}")
            return []

    async def label_frame_with_box(
        self,
        image_path: Path,
        box: tuple[int, int, int, int],  # (x1, y1, x2, y2)
        class_id: int = 0,
    ) -> list[dict]:
        """
        Refine a bounding box using SAM3.

        Args:
            image_path: Path to frame image
            box: Bounding box as (x1, y1, x2, y2)
            class_id: Class ID to assign

        Returns:
            List of refined detections
        """
        await self.load_model()

        if self.predictor is None:
            return []

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        try:
            self.predictor.set_image(str(image_path))

            # SAM3 box prompt
            results = self.predictor(bboxes=[list(box)])

            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    for mask in masks:
                        bbox = self._mask_to_bbox(mask, width, height)
                        if bbox:
                            det = {
                                "box": bbox,
                                "confidence": 1.0,
                                "class_id": class_id,
                            }
                            poly = self._mask_to_polygon_norm(mask, width, height)
                            if poly is not None:
                                det["polygon"] = poly
                            detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"SAM3 box prompt failed: {e}")
            return []

    async def _fallback_detect(
        self,
        image_path: Path,
        class_prompts: list[str],
    ) -> list[dict]:
        """Fallback detection using standard YOLO."""
        try:
            from ultralytics import YOLO

            model = YOLO("yolo11n.pt")
            results = model(str(image_path), verbose=False)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    # Get class name from COCO classes
                    coco_name = result.names.get(cls_id, f"class_{cls_id}")

                    # Try to match with user's class prompts
                    matched_class_id = 0
                    for idx, prompt in enumerate(class_prompts):
                        if (
                            prompt.lower() in coco_name.lower()
                            or coco_name.lower() in prompt.lower()
                        ):
                            matched_class_id = idx
                            break

                    # Convert to normalized xywh
                    img_h, img_w = result.orig_shape
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2 / img_w
                    cy = (y1 + y2) / 2 / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h

                    detections.append(
                        {
                            "box": {
                                "x": float(cx),
                                "y": float(cy),
                                "width": float(w),
                                "height": float(h),
                            },
                            "confidence": conf,
                            "class_id": matched_class_id,
                        }
                    )

            return detections

        except Exception as e:
            logger.error(f"Fallback detection failed: {e}")
            return []

    @staticmethod
    def _mask_to_bbox(
        mask: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> dict | None:
        """Convert binary mask to normalized bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return None

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        # Convert to normalized center format
        cx = (x1 + x2) / 2 / img_width
        cy = (y1 + y2) / 2 / img_height
        w = (x2 - x1) / img_width
        h = (y2 - y1) / img_height

        return {"x": float(cx), "y": float(cy), "width": float(w), "height": float(h)}

    @staticmethod
    def _mask_to_polygon_norm(
        mask: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> list[list[float]] | None:
        """Convert a binary mask to a simplified, normalised polygon ([[x,y], ...] in [0,1]).

        Uses the largest external contour simplified with cv2.approxPolyDP at 1.5px.
        Returns None for empty/degenerate masks.
        """
        try:
            arr = np.asarray(mask)
            if arr.ndim == 3:
                arr = arr.squeeze()
            if arr.ndim != 2:
                return None
            bin_mask = (arr > 0).astype(np.uint8)
            if bin_mask.max() == 0:
                return None
            contours, _ = cv2.findContours(
                bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            biggest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(biggest) <= 0:
                return None
            simplified = cv2.approxPolyDP(biggest, epsilon=1.5, closed=True)
            pts = simplified.reshape(-1, 2)
            if len(pts) < 3:
                return None
            return [[float(x) / img_width, float(y) / img_height] for x, y in pts]
        except Exception:
            return None


# Global instance
sam_labeler = SAMLabeler()
