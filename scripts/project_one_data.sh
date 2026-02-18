#!/usr/bin/env bash
# Replayable data setup: imports + class merges.
# Copy this script, set PROJECT and run the same steps on any new project.
# Requires: .env with ROBOFLOW_API_KEY (or export it).

set -e
PROJECT="${PROJECT:-data/projects/One}"
WORKSPACE="${WORKSPACE:-batman-mhu3h}"

echo "=== Data prep for project: $PROJECT ==="

# 1. Imports (create project first with --create if needed)
echo "Importing COCO Zoo: person..."
uv run python -m cli.importer coco \    
  --project "$PROJECT" \
  --create \
  --classes person \
  --split validation \
  --max-samples 300

echo "Importing Roboflow: crane hook..."
uv run python -m cli.importer roboflow \
  --project "$PROJECT" \
  --workspace "$WORKSPACE" \
  --rf-project crane-hook-vycl8 \
  --version 1

echo "Merging crane-hook classes into crane_hook..."
uv run python -m cli.classes merge \
  --project "$PROJECT" \
  --source "crane-hook" "crane hook" \
  --target "crane_hook"

echo "Importing Roboflow: spreader..."
uv run python -m cli.importer roboflow \
  --project "$PROJECT" \
  --workspace "$WORKSPACE" \
  --rf-project spreader-only-j6irz \
  --version 1

# Add more merge/rename steps as needed:
# uv run python -m cli.classes merge --project "$PROJECT" --source "spreader" "container-spreader" --target "spreader"
# uv run python -m cli.classes rename --project "$PROJECT" --old-name "old" --new-name "new"

echo "=== Data prep done for $PROJECT ==="
