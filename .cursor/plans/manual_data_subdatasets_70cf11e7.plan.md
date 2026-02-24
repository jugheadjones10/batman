---
name: Manual Data Subdatasets
overview: Extend the manual_data system to support named subdatasets (subdirectories within manual_data/) with the ability to include or exclude specific datasets during training.
todos:
  - id: sync-api
    content: Update manual_data sync to scan subdirectories, create per-dataset frames.json, add /datasets endpoint
    status: completed
  - id: trainer-categorize
    content: Update _categorize_frame_dir() and load_project_data() in trainer.py to recognize and filter manual subdatasets
    status: completed
  - id: prepare-coco
    content: Thread manual_datasets/exclude params through prepare_coco_dataset()
    status: completed
  - id: cli-flags
    content: Add --manual-datasets and --exclude-manual-datasets to cli/train.py
    status: completed
  - id: slurm-script
    content: Add manual dataset flags to submit_train.sh
    status: completed
  - id: api-models
    content: Add manual_datasets fields to DatasetExportConfig and update training API
    status: completed
  - id: dataset-exporter
    content: Update DatasetExporter to recognize manual_data__ prefixed video IDs as manual data
    status: completed
  - id: docs
    content: Update mkdocs with new manual dataset CLI options and directory structure
    status: completed
isProject: false
---

# Manual Data Subdatasets

## Current State

All manual images live in a flat `manual_data/` folder. A single `frames/manual_data/frames.json` tracks them with frame IDs like `manual_data_000000`. During training, manual data is either all-in or all-out via `--sources manual_data`.

## Proposed Design

### Directory Layout

Subdirectories inside `manual_data/` become named datasets. Root-level images remain as the "default" dataset for backward compatibility.

```
manual_data/
  img_root.jpg                  # -> "manual_data" (unchanged)
  crane_closeups/
    img1.jpg                    # -> "manual_data__crane_closeups"
  worker_shots/
    img2.jpg                    # -> "manual_data__worker_shots"

frames/
  manual_data/
    frames.json                 # IDs: manual_data_000000, ...
  manual_data__crane_closeups/
    frames.json                 # IDs: manual_data__crane_closeups_000000, ...
  manual_data__worker_shots/
    frames.json                 # IDs: manual_data__worker_shots_000000, ...
```

The double-underscore `__` separator clearly distinguishes subdataset names from the index suffix. Frame IDs like `manual_data__crane_closeups_000000` still pass the existing `startswith("manual_data")` checks, so all manual frames continue to be preserved during class-based sampling, and the manual split strategy logic works without changes.

### Selection Mechanism (CLI)

```bash
# Include all manual datasets (backward compatible)
python -m cli.train --project ... --sources manual_data,imports

# Include only specific manual datasets
python -m cli.train --project ... --sources manual_data,imports \
    --manual-datasets crane_closeups,worker_shots

# Exclude specific manual datasets
python -m cli.train --project ... --sources manual_data,imports \
    --exclude-manual-datasets negative_examples
```

`--manual-datasets` and `--exclude-manual-datasets` are mutually exclusive. They only apply when `manual_data` is in `--sources`.

## Files to Change

### 1. Sync API - [backend/app/api/manual_data.py](backend/app/api/manual_data.py)

- Update `sync_manual_data()` to scan subdirectories of `manual_data/`
- For each subdirectory `{name}/`, create a separate `frames/manual_data__{name}/frames.json` with frame IDs `manual_data__{name}_NNNNNN`
- Root-level images still go to `frames/manual_data/frames.json` (backward compat)
- Add a `GET /manual-data/datasets` endpoint returning the list of dataset names
- Update `list_manual_data_images()` to accept an optional `dataset` query param
- Update image serving to handle subdirectory paths

### 2. Source Categorization - [src/core/trainer.py](src/core/trainer.py)

- Update `_categorize_frame_dir()` (line 148):

```python
  def _categorize_frame_dir(dir_name, video_dir_names):
      if dir_name == "manual_data" or dir_name.startswith("manual_data__"):
          return "manual_data"
      ...


```

- Add `manual_datasets` and `exclude_manual_datasets` parameters to `load_project_data()` and `prepare_coco_dataset()`
- In the source-filtering branch of `load_project_data()` (line 206), apply dataset-level filtering:
  - If `manual_datasets` is set, only include `manual_data__X` dirs where `X` is in the list (plus root `manual_data` if "default" is in the list)
  - If `exclude_manual_datasets` is set, exclude those

### 3. CLI - [cli/train.py](cli/train.py)

- Add `--manual-datasets` argument (comma-separated dataset names)
- Add `--exclude-manual-datasets` argument (comma-separated, mutually exclusive with above)
- Pass through to `load_project_data()` and `prepare_coco_dataset()`
- Record in `training_config.json`

### 4. SLURM Script - [submit_train.sh](submit_train.sh)

- Add `--manual-datasets=NAMES` argument
- Add `--exclude-manual-datasets=NAMES` argument
- Pass through to the `python -m cli.train` command

### 5. API Models - [backend/app/models/training.py](backend/app/models/training.py)

- Add `manual_datasets: Optional[list[str]]` and `exclude_manual_datasets: Optional[list[str]]` to `DatasetExportConfig`

### 6. Training API - [backend/app/api/training.py](backend/app/api/training.py)

- In `export_dataset()`, apply manual dataset filtering when loading frames (similar logic to `load_project_data`)

### 7. Dataset Exporter - [backend/app/services/dataset_exporter.py](backend/app/services/dataset_exporter.py)

- Update the manual frame check from `vid_id == "manual_data"` to also match `manual_data__*` prefixed video IDs

### 8. Frontend (ProjectPage) - follow-up

- Show manual datasets grouped by subdirectory name
- Allow selecting/deselecting datasets in training config UI
- This is a larger change and can be a separate follow-up

## Backward Compatibility

- Root-level images in `manual_data/` work exactly as before
- `--sources manual_data` without `--manual-datasets` includes ALL manual datasets (root + all subdirectories)
- Existing frame IDs and annotations are untouched
- The `startswith("manual_data")` checks used for sampling preservation and split strategies continue to match all manual frame IDs
