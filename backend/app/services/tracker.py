"""Object tracking service for linking detections across frames."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from loguru import logger


@dataclass
class TrackingConfig:
    """Tracking configuration."""

    mode: Literal["visible_only", "occlusion_tolerant"] = "visible_only"
    max_age: int = 30  # Frames before track is lost
    iou_threshold: float = 0.3  # Minimum IoU for association
    min_hits: int = 3  # Minimum detections to confirm track
    use_appearance_embedding: bool = False

    @classmethod
    def visible_only(cls) -> "TrackingConfig":
        """Conservative tracking - tracks end quickly when object disappears."""
        return cls(
            mode="visible_only",
            max_age=5,
            iou_threshold=0.4,
            min_hits=3,
        )

    @classmethod
    def occlusion_tolerant(cls) -> "TrackingConfig":
        """Tolerant tracking - allows re-linking after disappearance."""
        return cls(
            mode="occlusion_tolerant",
            max_age=30,
            iou_threshold=0.3,
            min_hits=2,
        )


@dataclass
class Track:
    """Represents a single object track."""

    track_id: int
    class_id: int
    boxes: dict = field(default_factory=dict)  # frame_number -> box
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    confirmed: bool = False


class Tracker:
    """
    Simple IoU-based tracker for linking detections across frames.

    Uses Hungarian algorithm for optimal assignment.
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()
        self.tracks: list[Track] = []
        self.next_track_id = 1

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 1

    def update(
        self,
        detections: list[dict],
        frame_number: int,
    ) -> list[dict]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detections with 'box', 'class_id', 'confidence'
            frame_number: Current frame number

        Returns:
            Detections with assigned track_id
        """
        # Age all tracks
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        if not detections:
            # Remove dead tracks
            self._remove_dead_tracks()
            return []

        # Build cost matrix based on IoU
        cost_matrix = self._build_cost_matrix(detections)

        # Solve assignment problem
        if len(self.tracks) > 0 and len(detections) > 0:
            matched, unmatched_dets, unmatched_tracks = self._associate(
                cost_matrix, detections
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))

        # Update matched tracks
        for det_idx, track_idx in matched:
            self._update_track(self.tracks[track_idx], detections[det_idx], frame_number)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], frame_number)

        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1

        # Remove dead tracks
        self._remove_dead_tracks()

        # Assign track IDs to detections
        results = []
        for det in detections:
            det_box = det["box"]
            best_track = None
            best_iou = 0

            for track in self.tracks:
                if frame_number in track.boxes:
                    iou = self._iou(det_box, track.boxes[frame_number])
                    if iou > best_iou:
                        best_iou = iou
                        best_track = track

            if best_track and best_track.confirmed:
                det["track_id"] = best_track.track_id
            else:
                det["track_id"] = None

            results.append(det)

        return results

    def _build_cost_matrix(self, detections: list[dict]) -> np.ndarray:
        """Build cost matrix based on IoU between tracks and detections."""
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        cost_matrix = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(self.tracks):
            if not track.boxes:
                continue
            last_box = list(track.boxes.values())[-1]

            for j, det in enumerate(detections):
                iou = self._iou(last_box, det["box"])
                # Convert IoU to cost (1 - IoU)
                cost_matrix[i, j] = 1 - iou

                # Penalize class mismatch
                if track.class_id != det.get("class_id", 0):
                    cost_matrix[i, j] += 0.5

        return cost_matrix

    def _associate(
        self,
        cost_matrix: np.ndarray,
        detections: list[dict],
    ) -> tuple[list, list, list]:
        """Associate detections to tracks using Hungarian algorithm."""
        from scipy.optimize import linear_sum_assignment

        n_tracks, n_dets = cost_matrix.shape

        # Use Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(range(n_tracks))

        for row, col in zip(row_indices, col_indices):
            # Check if cost is below threshold (IoU > threshold)
            if cost_matrix[row, col] < (1 - self.config.iou_threshold):
                matched.append((col, row))  # (det_idx, track_idx)
                if col in unmatched_dets:
                    unmatched_dets.remove(col)
                if row in unmatched_tracks:
                    unmatched_tracks.remove(row)

        return matched, unmatched_dets, unmatched_tracks

    def _update_track(self, track: Track, detection: dict, frame_number: int):
        """Update track with new detection."""
        track.boxes[frame_number] = detection["box"]
        track.hits += 1
        track.time_since_update = 0

        if track.hits >= self.config.min_hits:
            track.confirmed = True

    def _create_track(self, detection: dict, frame_number: int):
        """Create new track from detection."""
        track = Track(
            track_id=self.next_track_id,
            class_id=detection.get("class_id", 0),
            boxes={frame_number: detection["box"]},
            hits=1,
            age=0,
            time_since_update=0,
        )
        self.next_track_id += 1
        self.tracks.append(track)

    def _remove_dead_tracks(self):
        """Remove tracks that haven't been updated for too long."""
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update < self.config.max_age
        ]

    @staticmethod
    def _iou(box1: dict, box2: dict) -> float:
        """Calculate IoU between two boxes in normalized center format."""
        # Convert to corner format
        x1_1 = box1["x"] - box1["width"] / 2
        y1_1 = box1["y"] - box1["height"] / 2
        x2_1 = box1["x"] + box1["width"] / 2
        y2_1 = box1["y"] + box1["height"] / 2

        x1_2 = box2["x"] - box2["width"] / 2
        y1_2 = box2["y"] - box2["height"] / 2
        x2_2 = box2["x"] + box2["width"] / 2
        y2_2 = box2["y"] + box2["height"] / 2

        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # Calculate union
        area1 = box1["width"] * box1["height"]
        area2 = box2["width"] * box2["height"]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_all_tracks(self) -> list[dict]:
        """Get all confirmed tracks."""
        return [
            {
                "track_id": t.track_id,
                "class_id": t.class_id,
                "boxes": t.boxes,
                "start_frame": min(t.boxes.keys()) if t.boxes else 0,
                "end_frame": max(t.boxes.keys()) if t.boxes else 0,
                "confirmed": t.confirmed,
            }
            for t in self.tracks
            if t.confirmed
        ]


def detect_problems(
    tracks: list[dict],
    frame_data: dict[int, list[dict]],
) -> list[dict]:
    """
    Detect potential labeling problems for the review queue.

    Args:
        tracks: List of track dictionaries
        frame_data: Mapping of frame_number to list of annotations

    Returns:
        List of problem items
    """
    problems = []

    for track in tracks:
        boxes = track.get("boxes", {})
        if len(boxes) < 2:
            continue

        sorted_frames = sorted(boxes.keys())

        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i - 1]
            curr_frame = sorted_frames[i]
            prev_box = boxes[prev_frame]
            curr_box = boxes[curr_frame]

            # Check for sudden position jumps
            pos_diff = np.sqrt(
                (curr_box["x"] - prev_box["x"]) ** 2 +
                (curr_box["y"] - prev_box["y"]) ** 2
            )
            if pos_diff > 0.2:  # More than 20% of image dimension
                problems.append({
                    "frame_number": curr_frame,
                    "problem_type": "box_jump",
                    "severity": min(pos_diff / 0.5, 1.0),
                    "description": f"Sudden position jump in track {track['track_id']}",
                    "affected_track_ids": [track["track_id"]],
                })

            # Check for sudden size changes
            size_ratio = (
                (curr_box["width"] * curr_box["height"]) /
                (prev_box["width"] * prev_box["height"] + 1e-6)
            )
            if size_ratio > 2.0 or size_ratio < 0.5:
                problems.append({
                    "frame_number": curr_frame,
                    "problem_type": "size_jump",
                    "severity": min(abs(np.log(size_ratio)) / 2, 1.0),
                    "description": f"Sudden size change in track {track['track_id']}",
                    "affected_track_ids": [track["track_id"]],
                })

            # Check for gaps (rapid appear/disappear)
            frame_gap = curr_frame - prev_frame
            if frame_gap > 10:  # Gap of more than 10 frames
                problems.append({
                    "frame_number": curr_frame,
                    "problem_type": "track_gap",
                    "severity": min(frame_gap / 50, 1.0),
                    "description": f"Track {track['track_id']} disappeared for {frame_gap} frames",
                    "affected_track_ids": [track["track_id"]],
                })

    # Sort by severity
    problems.sort(key=lambda x: -x["severity"])

    return problems

