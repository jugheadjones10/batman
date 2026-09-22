# Project folder structure

Batman stores project data under `data/projects/`, relative to the repository root.
Each project has its own videos, frames, labels, training runs, and inference results.

## Inside a project

This is a representative layout, including optional artifacts. Names in angle
brackets are placeholders. A folder or file may appear only after its workflow
has run; imported datasets and label snapshots are not present in every project.

```text
data/projects/<project_name>/
├── project.json
├── videos/
│   ├── videos.json
│   ├── <uploaded_video>.mp4                 (or .mkv, etc.)
│   ├── <video_id>_proxy.mp4
│   └── thumbnails/
├── frames/
│   ├── video_1/
│   │   ├── frames.json
│   │   └── video_1_000000.jpg               (and more frames)
│   └── <import_source_key>/
│       ├── frames.json
│       └── <frame_id>.jpg
├── imports/
│   └── imports.json
├── labels/
│   ├── current/
│   │   ├── annotations.json
│   │   └── tracks.json
│   └── iteration_<n>/
├── exports/
│   └── coco/
│       ├── train/
│       │   ├── _annotations.coco.json
│       │   └── <image>.jpg
│       ├── valid/
│       │   ├── _annotations.coco.json
│       │   └── <image>.jpg
│       └── test/
│           ├── _annotations.coco.json
│           └── <image>.jpg
├── runs/
│   └── <run_name>/
│       ├── meta.json
│       ├── training_config.json
│       ├── class_info.json
│       ├── results.json
│       ├── checkpoint_best_total.pth
│       ├── last.ckpt
│       ├── training.log
│       ├── metrics.csv
│       ├── hparams.yaml
│       └── events.out.tfevents.<...>
└── inference/
    └── <run_name>/
        └── <video_id>/
            └── <inference_id>/
                ├── result.json
                └── detected.mp4
```



## Folder comments: what goes where

All paths below are relative to `data/projects/<project_name>/`.


| Folder or file                                    | Contents and purpose                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project.json`                                    | Project name, description, classes, configuration, counts, and other project metadata.                                                                                                                                                                                                                                                                                       |
| `videos/`                                         | Uploaded source videos and generated playback proxies. `videos.json` connects video IDs to their files and metadata.                                                                                                                                                                                                                                                         |
| `videos/thumbnails/`                              | Generated preview images used when browsing videos.                                                                                                                                                                                                                                                                                                                          |
| `frames/`                                         | Still images extracted from videos or brought in by dataset imports. These are the images used for labeling and dataset preparation.                                                                                                                                                                                                                                         |
| `frames/<video_id>/`                              | Frames from one uploaded video. `frames.json` records frame metadata; JPEG files hold the actual images.                                                                                                                                                                                                                                                                     |
| `frames/<import_source_key>/`                     | Images from one imported source, such as `roboflow_crane-hook_1` or `coco_zoo_person_1`, together with `frames.json`. Legacy sources may use numeric folder names such as `1` or `-1`.                                                                                                                                                                                       |
| `imports/`                                        | Import provenance. `imports.json` records source details and identifiers; the imported images themselves live in `frames/`.                                                                                                                                                                             |
| `labels/`                                         | Editable annotations and any saved labeling iterations.                                                                                                                                                                                                                                                                                                                      |
| `labels/current/`                                 | Active labels in `annotations.json`, plus track metadata in `tracks.json` when present. Human corrections and automatic labeling update the active annotations.                                                                                                                                                                                                              |
| `labels/iteration_<n>/`                           | Saved snapshots of a labeling iteration, created by the iteration workflow.                                                                                                                                                                                                                                                                                                  |
| `exports/`                                        | Generated datasets prepared from project images and annotations. The default COCO export is `exports/coco/`.                                                                                                                                                                                                                                                                 |
| `exports/coco/train/`                             | Training images and their COCO annotations: the examples the model learns from.                                                                                                                                                                                                                                                                                              |
| `exports/coco/valid/`                             | Validation images and annotations used to evaluate progress during training. The directory is named `valid`, not `val`.                                                                                                                                                                                                                                                      |
| `exports/coco/test/`                              | Held-out test images and annotations for evaluation. Split contents depend on export settings.                                                                                                                                                                                                                                                                               |
| `runs/`                                           | Training outputs, grouped into a separate folder for each run.                                                                                                                                                                                                                                                                                                               |
| `runs/<run_name>/`                                | Model checkpoints, training settings, class information, metrics, and logs. `meta.json` records backend job status; `training_config.json` records the invocation; `class_info.json` records model classes; `results.json` contains evaluation results when produced. Checkpoint and log names vary by training backend and version; the tree shows representative examples. |
| `inference/`                                      | Saved predictions made by trained models on project videos.                                                                                                                                                                                                                                                                                                                  |
| `inference/<run_name>/`                           | Results associated with a particular training run. The name connects these predictions to `runs/<run_name>/`.                                                                                                                                                                                                                                                                |
| `inference/<run_name>/<video_id>/`                | Results for one video, potentially containing several inference sessions.                                                                                                                                                                                                                                                                                                    |
| `inference/<run_name>/<video_id>/<inference_id>/` | One timestamped inference session, such as `20260901_144813`. `result.json` stores settings, statistics, and per-frame predictions; optional `detected.mp4` shows predictions overlaid on the video. Saved distance calibration is stored inside `result.json` as `z_calibration`.                                                                                           |




### Practical notes

- The usual flow is **videos/imports → frames → labels → exports → training runs → inference**.
- Existing projects can have empty or missing workflow folders. For example,
`Phase 2 Z height` currently has no `inference/` folder and no `labels/current/`.
- Older inference results can live directly at
`inference/<run_name>/<video_id>/result.json`, without an inference-session folder.
- Keep metadata together with the media it describes. Adding an image or video
file alone does not register it with the application; use the upload or import workflow.
- This guide describes project-local storage. Repository-level folders such as
`datasets/` and `runs/` are separate locations used by some standalone workflows
or explicit output-path overrides.

For workflow details, see [dataset imports](../cli/importer.md),
[training](training.md), and [inference](inference.md).