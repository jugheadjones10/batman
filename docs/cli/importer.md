# Data Importer CLI

Import datasets from Roboflow or COCO Zoo into Batman projects.

## Overview

The importer CLI supports:
- **Roboflow datasets**: Import via API with workspace/project/version
- **COCO Zoo**: Import specific classes via FiftyOne
- **Project creation**: Optionally create projects during import
- **Class mapping**: Automatic class name extraction
- **Source tracking**: Detailed metadata about import source is stored with each frame

## Basic Usage

```bash
# Import from Roboflow
python -m cli.importer roboflow \
  --project data/projects/MyProject \
  --workspace myworkspace \
  --rf-project myproject \
  --version 1

# Import from COCO Zoo
python -m cli.importer coco \
  --project data/projects/MyProject \
  --classes person car bicycle
```

## Subcommands

### `roboflow` - Import from Roboflow

Import a Roboflow dataset via API.

#### Parameters

##### `--project PATH` (Required)
Batman project path.

##### `--create`
Create project if it doesn't exist.

##### `--api-key KEY`
Roboflow API key. Can also use `ROBOFLOW_API_KEY` environment variable.

##### `--workspace NAME` (Required)
Roboflow workspace name.

##### `--rf-project NAME` (Required)
Roboflow project name.

##### `--version N` (Required)
Dataset version number.

##### `--format FORMAT`
Download format.
- **Default**: `coco`

#### Example

```bash
# With API key as argument
python -m cli.importer roboflow \
  --project data/projects/CraneHook \
  --create \
  --api-key YOUR_API_KEY \
  --workspace myworkspace \
  --rf-project crane-detection \
  --version 1

# With environment variable
export ROBOFLOW_API_KEY=YOUR_API_KEY
python -m cli.importer roboflow \
  --project data/projects/CraneHook \
  --create \
  --workspace myworkspace \
  --rf-project crane-detection \
  --version 1
```

### `coco` - Import from COCO Zoo

Import specific classes from COCO via FiftyOne.

#### Parameters

##### `--project PATH` (Required)
Batman project path.

##### `--create`
Create project if it doesn't exist.

##### `--classes CLASS [CLASS ...]` (Required)
COCO class names to import (one or more).

##### `--split SPLIT`
Dataset split to import.
- **Choices**: `train`, `validation`, `test`
- **Default**: `validation`

##### `--max-samples N`
Maximum number of samples to import.
- **Default**: All samples

#### Example

```bash
# Import person and car classes
python -m cli.importer coco \
  --project data/projects/PersonCar \
  --create \
  --classes person car

# Import with limits
python -m cli.importer coco \
  --project data/projects/Test \
  --create \
  --classes person car bicycle \
  --split train \
  --max-samples 100
```

### `list` - List Projects

List all Batman projects.

#### Parameters

##### `--projects-dir PATH`
Projects directory.
- **Default**: `data/projects`

#### Example

```bash
python -m cli.importer list

# Custom projects directory
python -m cli.importer list --projects-dir /path/to/projects
```

## COCO Class Names

Common COCO classes you can import:

### People & Animals
- `person`
- `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`
- `bird`

### Vehicles
- `bicycle`, `car`, `motorcycle`, `airplane`, `bus`, `train`, `truck`, `boat`

### Traffic & Street
- `traffic light`, `fire hydrant`, `stop sign`, `parking meter`, `bench`

### Indoor Objects
- `chair`, `couch`, `potted plant`, `bed`, `dining table`, `toilet`
- `tv`, `laptop`, `mouse`, `remote`, `keyboard`, `cell phone`
- `microwave`, `oven`, `toaster`, `sink`, `refrigerator`
- `book`, `clock`, `vase`, `scissors`

### Kitchen
- `bottle`, `wine glass`, `cup`, `fork`, `knife`, `spoon`, `bowl`

### Sports
- `frisbee`, `skis`, `snowboard`, `sports ball`, `kite`
- `baseball bat`, `baseball glove`, `skateboard`, `surfboard`, `tennis racket`

### Accessories
- `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`

[Full list of 80 COCO classes](https://cocodataset.org/#explore)

## Examples

### Example 1: Create New Project from COCO

```bash
python -m cli.importer coco \
  --project data/projects/VehicleDetection \
  --create \
  --classes car truck bus motorcycle bicycle
```

### Example 2: Import Roboflow Dataset

```bash
export ROBOFLOW_API_KEY=your_api_key_here

python -m cli.importer roboflow \
  --project data/projects/CustomDataset \
  --create \
  --workspace your-workspace \
  --rf-project your-project \
  --version 2
```

### Example 3: Limited COCO Import

Import only 50 samples for quick testing:

```bash
python -m cli.importer coco \
  --project data/projects/QuickTest \
  --create \
  --classes person \
  --max-samples 50 \
  --split validation
```

### Example 4: Training Set Import

Import from COCO training set:

```bash
python -m cli.importer coco \
  --project data/projects/LargeDataset \
  --create \
  --classes person car bicycle \
  --split train \
  --max-samples 1000
```

### Example 5: List Projects

```bash
# List all projects
python -m cli.importer list

# List from custom directory
python -m cli.importer list --projects-dir /mnt/data/projects
```

## Project Structure

After import, projects have this structure:

```
data/projects/MyProject/
├── project.json        # Project metadata
├── imports/           # Import tracking
│   └── imports.json   # Metadata for all imports (includes video_id for each)
├── videos/            # Empty (for later video uploads)
├── frames/            # All frames (videos and imports)
│   ├── -1/           # First import (unique video_id)
│   │   ├── frames.json      # Frame metadata with import_id references
│   │   ├── -1000000.jpg
│   │   ├── -1000001.jpg
│   │   └── ...
│   ├── -2/           # Second import (unique video_id)
│   │   ├── frames.json
│   │   └── *.jpg
│   ├── -3/           # Third import
│   ├── 1/            # Real video #1 (positive IDs)
│   └── 2/            # Real video #2
└── labels/            # Imported annotations
    └── current/
        └── annotations.json
```

**Important**: Each import gets its own unique negative video_id (-1, -2, -3, ...). This keeps imports separate and makes them easy to manage individually.

### project.json

```json
{
  "name": "MyProject",
  "classes": ["person", "car", "bicycle"],
  "created_at": "2026-01-28T10:30:00",
  "data_source": "coco_validation",
  "num_frames": 150
}
```

## Source Metadata Tracking

When importing data, the importer automatically stores detailed information about where the data came from. To avoid duplicating metadata across thousands of frames, the system uses an efficient referencing approach:

1. **Import metadata is stored once** in `imports/imports.json`
2. **Each frame references the import** via an `import_id`

This keeps your project files small and efficient while maintaining full data provenance.

### Import Metadata Storage

All imports are tracked in `data/projects/MyProject/imports/imports.json`:

```json
{
  "import_1_20260128_103000": {
    "type": "roboflow",
    "workspace": "myworkspace",
    "project": "crane-detection",
    "version": 1,
    "format": "coco",
    "imported_at": "2026-01-28T10:30:00.123456"
  },
  "import_2_20260128_110000": {
    "type": "coco_zoo",
    "classes": ["person", "car"],
    "split": "validation",
    "max_samples": 100,
    "imported_at": "2026-01-28T11:00:00.123456"
  },
  "import_3_20260128_120000": {
    "type": "local_coco",
    "path": "/path/to/dataset",
    "imported_at": "2026-01-28T12:00:00.123456"
  }
}
```

### Frame References

Each frame stores only a lightweight reference to the import:

```json
{
  "video_id": -1,
  "frame_number": 0,
  "source": "roboflow",
  "import_id": "import_1_20260128_103000",
  "original_filename": "image_001.jpg",
  "split": "train"
}
```

### Benefits

This design provides:
- **Efficiency**: Import metadata stored once, not duplicated per frame
- **Complete provenance**: Full details about data origin
- **Easy updates**: Change import metadata in one place
- **Query capability**: Find all frames from a specific import

### Accessing Source Metadata

You can access this metadata programmatically:

```python
from src.core.project import Project

project = Project.load("data/projects/MyProject")

# Load import metadata
imports_meta = project.load_imports_metadata()

# Load frames
frames_meta = project.load_frames_meta(video_id=-1)  # -1 for Roboflow imports

for frame_id, frame_data in frames_meta.items():
    import_id = frame_data.get("import_id")
    if import_id and import_id in imports_meta:
        import_info = imports_meta[import_id]
        print(f"Frame {frame_id} imported from: {import_info['type']}")
        
        if import_info["type"] == "roboflow":
            print(f"  Workspace: {import_info['workspace']}")
            print(f"  Project: {import_info['project']}")
            print(f"  Version: {import_info['version']}")
```

### Example: Finding All Frames from a Specific Roboflow Version

```python
from src.core.project import Project

project = Project.load("data/projects/MyProject")
imports_meta = project.load_imports_metadata()

# Find imports from a specific Roboflow version
target_imports = [
    import_id for import_id, meta in imports_meta.items()
    if meta["type"] == "roboflow" 
    and meta["workspace"] == "myworkspace"
    and meta["project"] == "crane-detection"
    and meta["version"] == 1
]

# Find all frames from those imports
frames_meta = project.load_frames_meta(video_id=-1)
matching_frames = [
    frame_id for frame_id, frame in frames_meta.items()
    if frame.get("import_id") in target_imports
]

print(f"Found {len(matching_frames)} frames from myworkspace/crane-detection v1")
```

## Environment Variables

### ROBOFLOW_API_KEY

Set for Roboflow imports:

```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

Get your API key from: https://roboflow.com/account/api

## Tips

### 1. Start Small

Import small datasets for testing:

```bash
--max-samples 50
```

### 2. Use Validation Split

Validation split is smaller and faster:

```bash
--split validation
```

### 3. Check Class Names

COCO class names are case-sensitive and may have spaces:

```bash
--classes "traffic light" "stop sign"
```

### 4. Create Projects Proactively

Use `--create` to avoid manual project creation:

```bash
python -m cli.importer coco --project data/projects/New --create --classes person
```

### 5. Verify Import

After import, check the project:

```bash
python -m cli.importer list
python -m cli.classes list --project data/projects/MyProject
```

## Troubleshooting

### Roboflow API Key Error

```bash
export ROBOFLOW_API_KEY=your_key
# or
--api-key your_key
```

### Class Not Found in COCO

Check class name spelling:

```bash
# Wrong
--classes "human"

# Correct
--classes "person"
```

### FiftyOne Download Issues

FiftyOne may need to download dataset first time (large download):

```bash
# Be patient on first run
python -m cli.importer coco --project ... --classes person
```

### Project Already Exists

Use existing project or choose different name:

```bash
# Use existing
python -m cli.importer coco --project data/projects/Existing --classes car

# Or create new
python -m cli.importer coco --project data/projects/New --create --classes car
```

## Related

- **[Training CLI](train.md)** - Train on imported data
- **[Class Management CLI](classes.md)** - Manage classes
- **[Training Workflow Guide](../guides/training.md)** - Complete workflow
