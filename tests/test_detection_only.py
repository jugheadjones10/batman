"""Detection-only contracts: reject obsolete inputs and preserve box workflows."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.models.annotation import AnnotationCreate, AnnotationUpdate
from backend.app.models.training import LocalTrainingSubmitRequest, TrainingSubmitRequest
from backend.app.services.dataset_exporter import DatasetExporter
from backend.app.services.inference_runner import InferenceRunner
from backend.app.services.sam_worker import run_one
from src.core.inference import RFDETRInference
from src.core.trainer import create_coco_split, resolve_rfdetr_class, validate_detection_checkpoint


BOX = {"x": 0.5, "y": 0.5, "width": 0.5, "height": 0.5}
POLYGON = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75]]


@pytest.mark.parametrize("request_model", [LocalTrainingSubmitRequest, TrainingSubmitRequest])
def test_training_accepts_detection_sizes_and_rejects_obsolete_options(request_model):
    for size in ("nano", "small", "base", "medium", "large"):
        request = request_model(training={"model": size})
        assert request.training.model == size
        assert "task" not in request.training.model_dump()
    for config in ({"task": "segmentation"}, {"model": "xlarge"}):
        with pytest.raises(ValidationError):
            request_model(training=config)


def test_polygon_annotation_requests_are_rejected():
    assert AnnotationCreate(frame_id=1, class_label_id=0, box=BOX).box.width == 0.5
    with pytest.raises(ValidationError):
        AnnotationCreate(frame_id=1, class_label_id=0, box=BOX, polygon=POLYGON)
    with pytest.raises(ValidationError):
        AnnotationUpdate(polygon=POLYGON)


def test_non_detection_model_size_is_rejected_before_import():
    with pytest.raises(ValueError, match="Unsupported detection model size"):
        resolve_rfdetr_class("xlarge")


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_checkpoint_metadata_is_rejected(tmp_path, nested):
    weights_dir = tmp_path / "weights" if nested else tmp_path
    weights_dir.mkdir(exist_ok=True)
    checkpoint = weights_dir / "best.pth"
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps({"config": {"training": {"task": "segmentation"}}}))
    with pytest.raises(ValueError, match="Only detection checkpoints"):
        validate_detection_checkpoint(checkpoint)
    metadata.write_text(json.dumps({"config": {"training": {"task": "detection"}}}))
    validate_detection_checkpoint(checkpoint)


def test_both_coco_exporters_use_existing_boxes_without_masks(tmp_path):
    image = tmp_path / "source.jpg"
    Image.new("RGB", (100, 80)).save(image)
    annotation = {"frame_id": "1", "class_label_id": 0, **BOX, "polygon": POLYGON}
    output = tmp_path / "core"
    output.mkdir()
    assert create_coco_split(
        {"1"}, {"1": {"image_path": str(image)}}, {"1": annotation}, ["spreader"], output
    ) == (1, 1)
    backend_output = tmp_path / "backend"
    asyncio.run(DatasetExporter(tmp_path)._export_coco(
        backend_output, {"train": [{"id": "1", "image_path": str(image)}]},
        {"1": [annotation]}, ["spreader"],
    ))
    for directory in (output, backend_output / "train"):
        data = json.loads((directory / "_annotations.coco.json").read_text())
        assert data["annotations"][0]["bbox"] == [25, 20, 50, 40]
        assert "segmentation" not in data["annotations"][0]
        assert "polygon" not in data["annotations"][0]


def test_inference_parsers_produce_boxes_only():
    output = SimpleNamespace(
        xyxy=np.array([[25, 20, 75, 60]]), class_id=np.array([0]),
        confidence=np.array([0.9]), mask=np.ones((1, 80, 100)),
    )
    runner = InferenceRunner()
    runner.class_names = ["spreader"]
    parsed = runner._parse_rfdetr_results(output, (80, 100))
    assert parsed[0]["box"] == BOX
    assert "mask" not in parsed[0]
    engine = RFDETRInference(Path("unused.pth"), ["spreader"])
    engine.model = SimpleNamespace(predict=lambda *args, **kwargs: output)
    detection = engine.predict_image(Image.new("RGB", (100, 80)))[0]
    assert detection.bbox == (25, 20, 75, 60)
    assert not hasattr(detection, "mask")


def test_sam_mask_fallback_still_produces_bounding_boxes(tmp_path):
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 80)).save(image)
    mask = np.zeros((80, 100))
    mask[20:61, 25:76] = 1

    class Array:
        def cpu(self):
            return self

        def numpy(self):
            return np.array([mask])

    class Masks:
        data = Array()

        def __len__(self):
            return 1

    class Predictor:
        def set_image(self, path):
            pass

        def __call__(self, **kwargs):
            return [SimpleNamespace(boxes=None, masks=Masks())]

    assert run_one(Predictor(), image, ["spreader"]) == [
        {"box": BOX, "confidence": 1.0, "class_id": 0}
    ]


@pytest.mark.parametrize("args", [["--task", "segmentation"], ["--model", "xlarge"]])
def test_cli_rejects_removed_training_options(monkeypatch, args):
    from cli.train import main

    monkeypatch.setattr(sys, "argv", ["train", *args])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_local_and_gpu_launchers_build_detection_commands(tmp_path):
    from backend.app.api.training import _build_local_train_argv
    from backend.app.services.gpu_service import GPUService

    argv = _build_local_train_argv(tmp_path, tmp_path / "run", LocalTrainingSubmitRequest())
    assert argv[argv.index("--model") + 1] == "base"
    assert "--task" not in argv
    script, _ = GPUService().generate_training_script(
        project_dir=str(tmp_path), output_dir=str(tmp_path / "run"),
        output_dataset=str(tmp_path / "dataset"), model="base", epochs=1,
        batch_size=1, image_size=640, lr=1e-4, patience=10, grad_accum=1,
        gpu_type="a100-80", num_gpus=1, time_limit="01:00:00",
    )
    assert "--model base" in script
    assert "--task " not in script
