"""SAM3-based auto-labeling service using SAM3SemanticPredictor."""

from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from backend.app.config import settings


class SAMLabeler:
    """Auto-labeling service using SAM3SemanticPredictor for text-based segmentation."""

    def __init__(self):
        self.predictor = None
        self.device = settings.device
        self.model_path = Path("sam3.pt")  # Default to sam3.pt in project root
        self.current_image_path: Path | None = None

    async def load_model(self):
        """Load SAM3SemanticPredictor."""
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
                "half": True,  # Use FP16 for faster inference
                "save": False,  # Don't save results
                "device": self.device,
                "verbose": False,
            }

            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            logger.info(f"SAM3SemanticPredictor loaded successfully on {self.device}")

        except ImportError as e:
            logger.error(f"SAM3SemanticPredictor not available: {e}")
            self.predictor = None
        except Exception as e:
            logger.error(f"Failed to load SAM3: {e}")
            self.predictor = None

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

                                detections.append(
                                    {
                                        "box": {
                                            "x": float(cx),
                                            "y": float(cy),
                                            "width": float(w),
                                            "height": float(h),
                                        },
                                        "confidence": conf,
                                        "class_id": class_id,
                                    }
                                )

                        # Also extract from masks if no boxes
                        elif result.masks is not None and len(result.masks) > 0:
                            masks = result.masks.data.cpu().numpy()
                            for mask in masks:
                                bbox = self._mask_to_bbox(mask, width, height)
                                if bbox:
                                    detections.append(
                                        {
                                            "box": bbox,
                                            "confidence": 1.0,
                                            "class_id": class_id,
                                        }
                                    )

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
                            detections.append(
                                {
                                    "box": bbox,
                                    "confidence": 1.0,
                                    "class_id": class_id,
                                }
                            )

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
                            detections.append(
                                {
                                    "box": bbox,
                                    "confidence": 1.0,
                                    "class_id": class_id,
                                }
                            )

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


# Global instance
sam_labeler = SAMLabeler()
