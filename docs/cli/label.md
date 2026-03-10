# Label CLI (SAM3 Auto-Labeling)

Run SAM3 auto-labeling on a Batman project from the command line. Supports video frames and manual-data sources. Uses class descriptions (from the project or `--descriptions`) as text prompts for SAM3.

## Overview

The label CLI:

- Runs SAM3 semantic segmentation on selected frames
- Uses per-class descriptions as prompts (stored in project or overridden by `--descriptions`)
- Supports video frames (`--video`) or manual-data sources (`--source`)
- Can target all frames, specific frame IDs, or only unlabeled frames (`--skip-labeled`)

## Basic Usage

```bash
# Label all frames in a video
python -m cli.label --project data/projects/MyProject --video video_1

# Label only manual_data images (root dataset)
python -m cli.label --project data/projects/MyProject --source manual_data

# Label specific frames
python -m cli.label --project data/projects/MyProject --video video_1 --frames 0,5,10,20
```

## Parameters

### `--project PATH` (required)

Path to the Batman project directory. If not absolute, resolved under `data/projects/`.

### `--video ID`

Video ID or source key (e.g. `video_1`). When set, only frames from this video are considered. Mutually exclusive with `--source`.

### `--source KEY`

Source key to filter by (e.g. `manual_data`, `manual_data__mydataset`). Use this to run on manual-data images. Mutually exclusive with `--video`.

### `--frames IDLIST`

Comma-separated frame IDs. If omitted, all frames for the given `--video` or `--source` are used. When used with `--video` or `--source`, only IDs that belong to that source are processed.

Examples:

- Numeric: `0,5,10,20`
- String IDs: `video_1_000000,video_1_000005`

### `--descriptions JSON`

JSON object mapping class name to description (SAM prompt). Overrides project `class_descriptions` for this run. Project descriptions are still used for classes not listed here.

```bash
--descriptions '{"hook":"yellow metal crane hook","person":"construction worker in hard hat"}'
```

### `--confidence FLOAT`

Confidence threshold for SAM (default: `0.25`).

### `--skip-labeled` / `--no-skip-labeled`

- **`--skip-labeled`** (default): Skip frames that already have at least one annotation.
- **`--no-skip-labeled`**: Run on all selected frames regardless of existing labels.

## Examples

### Label all frames in a video

```bash
python -m cli.label --project data/projects/CraneHook --video video_1
```

### Label only manual_data images

```bash
python -m cli.label --project data/projects/MyProject --source manual_data
```

### Label a specific manual subdataset

```bash
python -m cli.label --project data/projects/MyProject --source manual_data__closeups
```

### Label specific frames with custom descriptions

```bash
python -m cli.label \
  --project data/projects/MyProject \
  --video video_1 \
  --frames 0,10,20,30 \
  --descriptions '{"crane_hook":"yellow metal hook suspended from cable"}'
```

### Re-label all frames (including already labeled)

```bash
python -m cli.label --project data/projects/MyProject --video video_1 --no-skip-labeled
```

## Class descriptions

Class descriptions are stored in the project and used as SAM3 text prompts. You can:

- Set them in the **web UI**: Annotate → Auto-label with SAM3 → edit the "Class descriptions" fields, then Run (they are saved to the project).
- Set them via **API**: `PUT /projects/{name}/class-descriptions` with a `dict[str, str]`.
- Override for a single run with **`--descriptions`** in the CLI.

If no description is set for a class, the class name itself is used as the prompt.

## Output

The CLI prints how many annotations were created and updates the project’s `labels/current/annotations.json`. Project-level `annotation_count` is updated when annotations are saved.

## Related

- **[Training CLI](train.md)** - Train models on labeled data
- **[Training Workflow](../guides/training.md)** - End-to-end workflow including annotation
- **[Videos CLI](videos.md)** - Manage videos and extract frames
