# Class Management CLI

List, rename, and merge object classes in Batman projects.

## Overview

The class management CLI allows you to:
- **List classes**: View all classes in a project
- **Rename classes**: Change class names
- **Merge classes**: Combine multiple classes into one

## Basic Usage

```bash
# List classes
python -m cli.classes list --project data/projects/MyProject

# Rename a class
python -m cli.classes rename \
  --project data/projects/MyProject \
  --old-name "crane-hook" \
  --new-name "crane_hook"

# Merge classes
python -m cli.classes merge \
  --project data/projects/MyProject \
  --source "crane-hook" "crane_hook" \
  --target "crane_hook"
```

## Subcommands

### `list` - List Classes

List all classes in a project with counts.

#### Parameters

##### `--project PATH` (Required)
Batman project path.

#### Example

```bash
python -m cli.classes list --project data/projects/MyProject
```

#### Output

```
Classes in project 'MyProject':
  1. person (245 annotations)
  2. car (189 annotations)
  3. bicycle (67 annotations)

Total: 3 classes, 501 annotations
```

### `rename` - Rename Class

Rename a single class across all annotations.

#### Parameters

##### `--project PATH` (Required)
Batman project path.

##### `--old-name NAME` (Required)
Current class name.

##### `--new-name NAME` (Required)
New class name.

#### Example

```bash
python -m cli.classes rename \
  --project data/projects/MyProject \
  --old-name "crane-hook" \
  --new-name "crane_hook"
```

#### Output

```
Renaming 'crane-hook' → 'crane_hook' in project 'MyProject'
Updated 45 annotations
Updated project.json
Done!
```

### `merge` - Merge Classes

Merge multiple classes into a single target class.

#### Parameters

##### `--project PATH` (Required)
Batman project path.

##### `--source NAME [NAME ...]` (Required)
Source class names to merge (one or more).

##### `--target NAME` (Required)
Target class name.

#### Example

```bash
python -m cli.classes merge \
  --project data/projects/MyProject \
  --source "crane-hook" "crane_hook" "hook" \
  --target "crane_hook"
```

#### Output

```
Merging classes in project 'MyProject':
  'crane-hook' → 'crane_hook' (23 annotations)
  'crane_hook' → 'crane_hook' (already target)
  'hook' → 'crane_hook' (15 annotations)

Updated 38 annotations
Updated project.json (removed duplicate classes)
Done!
```

## Use Cases

### 1. Standardize Class Names

Fix inconsistent naming from imports or manual labeling:

```bash
# Before: "crane-hook", "crane_hook", "hook"
# After: "crane_hook"

python -m cli.classes merge \
  --project data/projects/CraneDetection \
  --source "crane-hook" "hook" \
  --target "crane_hook"
```

### 2. Simplify Class Hierarchy

Merge fine-grained classes into broader categories:

```bash
# Merge vehicle types
python -m cli.classes merge \
  --project data/projects/Traffic \
  --source "sedan" "suv" "van" \
  --target "car"
```

### 3. Fix Typos

Correct misspelled class names:

```bash
python -m cli.classes rename \
  --project data/projects/MyProject \
  --old-name "persn" \
  --new-name "person"
```

### 4. Prepare for Training

Consolidate classes before training:

```bash
# List classes
python -m cli.classes list --project data/projects/MyProject

# Merge similar classes
python -m cli.classes merge \
  --project data/projects/MyProject \
  --source "pedestrian" "walker" \
  --target "person"

# Train with clean classes
python -m cli.train --project data/projects/MyProject
```

## Examples

### Example 1: Inspect Project

```bash
python -m cli.classes list --project data/projects/CraneHook
```

Output:
```
Classes in project 'CraneHook':
  1. crane-hook (150 annotations)
  2. crane_hook (45 annotations)
  3. hook (30 annotations)
  4. crane_boom (200 annotations)

Total: 4 classes, 425 annotations
```

### Example 2: Merge Variants

```bash
python -m cli.classes merge \
  --project data/projects/CraneHook \
  --source "crane-hook" "hook" \
  --target "crane_hook"
```

### Example 3: Rename for Consistency

```bash
python -m cli.classes rename \
  --project data/projects/CraneHook \
  --old-name "crane-boom" \
  --new-name "crane_boom"
```

### Example 4: Verify Changes

```bash
python -m cli.classes list --project data/projects/CraneHook
```

Output:
```
Classes in project 'CraneHook':
  1. crane_hook (225 annotations)
  2. crane_boom (200 annotations)

Total: 2 classes, 425 annotations
```

## Important Notes

### Irreversible Operations

Class renaming and merging modify project data:

!!! warning
    These operations are **irreversible**. Back up your project before making changes:
    ```bash
    cp -r data/projects/MyProject data/projects/MyProject.backup
    ```

### Case Sensitivity

Class names are case-sensitive:

```bash
# These are different classes:
"Person" != "person"
```

### Whitespace Handling

Leading/trailing whitespace is automatically stripped:

```bash
# These become the same:
" person " → "person"
"person" → "person"
```

### Effect on Annotations

- **Rename**: Updates all annotations with old name
- **Merge**: Updates all source annotations to target name
- **Project file**: Updates class list in `project.json`

## Best Practices

### 1. List Before Modifying

Always list classes first:

```bash
python -m cli.classes list --project data/projects/MyProject
```

### 2. Back Up Projects

```bash
cp -r data/projects/MyProject data/projects/MyProject.backup
```

### 3. Use Consistent Naming

Choose a naming convention:
- **snake_case**: `crane_hook`, `crane_boom`
- **kebab-case**: `crane-hook`, `crane-boom`
- **camelCase**: `craneHook`, `craneBoom`

Stick to one convention across all projects.

### 4. Merge Before Training

Clean up classes before training:

```bash
# Clean classes
python -m cli.classes merge --project ... --source ... --target ...

# Then train
python -m cli.train --project ...
```

### 5. Document Changes

Keep a log of class modifications:

```bash
# Log changes
echo "$(date): Merged crane-hook variants → crane_hook" >> data/projects/MyProject/changelog.txt
```

## Related

- **[Importer CLI](importer.md)** - Import datasets
- **[Training CLI](train.md)** - Train models
- **[Training Workflow Guide](../guides/training.md)** - Complete workflow
