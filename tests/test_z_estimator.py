import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

z_estimator = importlib.import_module("backend.app.services.z_estimator")


VIDEO_RESOLUTION = {"width": 1000, "height": 1000}
MODEL = {"type": "k_over_s", "k": 1000.0}


def _det(
    class_name: str,
    x: float,
    y: float,
    z_mm: float | None = None,
    width: float = 0.1,
    height: float = 0.1,
) -> dict:
    det = {
        "class_name": class_name,
        "confidence": 0.9,
        "box": {"x": x, "y": y, "width": width, "height": height},
    }
    if z_mm is not None:
        det["z_mm"] = z_mm
    return det


def test_estimate_selects_only_center_container_target() -> None:
    left = _det("container", 0.2, 0.5, z_mm=999.0)
    center = _det("container", 0.52, 0.49)
    right = _det("container", 0.8, 0.5, z_mm=999.0)
    spreader_a = _det("spreader", 0.4, 0.5)
    spreader_b = _det("spreader", 0.6, 0.5)
    frames = [
        {
            "frame_number": 1,
            "detections": [left, center, right, spreader_a, spreader_b],
        }
    ]

    result = z_estimator.estimate(
        MODEL,
        frames,
        VIDEO_RESOLUTION,
        target_classes=["spreader", "container"],
        reference_class="spreader",
    )

    detections = result[0]["detections"]
    assert detections[1]["z_mm"] == 10.0
    assert detections[1]["z_selected"] is True
    assert "z_mm" not in detections[0]
    assert "z_selected" not in detections[0]
    assert "z_mm" not in detections[2]
    assert "z_selected" not in detections[2]
    assert detections[3]["z_mm"] == 10.0
    assert detections[4]["z_mm"] == 10.0


def test_reference_container_preserves_all_detections_behavior() -> None:
    frames = [
        {
            "frame_number": 1,
            "detections": [
                _det("container", 0.2, 0.5),
                _det("container", 0.52, 0.49),
                _det("container", 0.8, 0.5),
            ],
        }
    ]

    result = z_estimator.estimate(
        MODEL,
        frames,
        VIDEO_RESOLUTION,
        target_classes=["container"],
        reference_class="container",
    )

    assert [det["z_mm"] for det in result[0]["detections"]] == [10.0, 10.0, 10.0]
    assert all("z_selected" not in det for det in result[0]["detections"])


def test_round_feature_calibration_uses_equivalent_size_ratio() -> None:
    frames = [
        {
            "frame_number": 1,
            "detections": [_det("spreader_round", 0.5, 0.5, width=0.02, height=0.02)],
        }
    ]

    model = z_estimator.calibrate(
        [{"frame_number": 1, "z_mm": 5.0, "detection_index": 0}],
        frames,
        VIDEO_RESOLUTION,
        reference_class="spreader_round",
        measurement_source=z_estimator.ROUND_FEATURE_EQUIVALENT_LENGTH,
        equivalent_size_ratio=10.0,
    )

    # Round feature diameter is 20 px; equivalent spreader size is 200 px.
    assert model == {"type": "k_over_s", "k": 1000.0}


def test_round_feature_estimate_scales_reference_but_not_container() -> None:
    frames = [
        {
            "frame_number": 1,
            "detections": [
                _det("spreader_round", 0.5, 0.4, width=0.02, height=0.02),
                _det("container", 0.5, 0.5, width=0.1, height=0.1),
            ],
        }
    ]

    result = z_estimator.estimate(
        MODEL,
        frames,
        VIDEO_RESOLUTION,
        target_classes=["spreader_round", "container"],
        reference_class="spreader_round",
        measurement_source=z_estimator.ROUND_FEATURE_EQUIVALENT_LENGTH,
        equivalent_size_ratio=10.0,
    )

    detections = result[0]["detections"]
    assert detections[0]["z_mm"] == 5.0
    assert detections[1]["z_mm"] == 10.0
    assert detections[1]["z_selected"] is True


def test_apply_round_feature_calibration_rejects_missing_ratio(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "z_calibration": {
                    "labels": [{"frame_number": 1, "z_mm": 5.0, "detection_index": 0}],
                    "reference_class": "spreader_round",
                    "targets": ["spreader_round"],
                    "video_resolution": VIDEO_RESOLUTION,
                    "measurement_source": z_estimator.ROUND_FEATURE_EQUIVALENT_LENGTH,
                },
                "frames": [
                    {
                        "frame_number": 1,
                        "detections": [
                            _det("spreader_round", 0.5, 0.5, width=0.02, height=0.02)
                        ],
                    }
                ],
            }
        )
    )

    try:
        z_estimator.apply_z_to_result(tmp_path)
    except ValueError as exc:
        assert "equivalent_size_ratio" in str(exc)
    else:
        raise AssertionError("Expected missing equivalent_size_ratio to be rejected")
