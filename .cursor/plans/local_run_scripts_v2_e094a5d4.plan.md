---
name: Local run scripts v2
overview: Add pre-job `manual_data/` + `project.json` sync to `run_inference.sh`, create a new `run_training.sh` with the same local runner pattern, and sync back only JSON metadata from training runs.
todos:
  - id: update-run-inference
    content: Add pre-sync (manual_data + project.json) and --no-push flag to run_inference.sh
    status: completed
  - id: create-run-training
    content: Create run_training.sh -- local training runner with pre-sync, SSH sbatch, and lightweight JSON-only post-sync
    status: completed
  - id: update-docs-v2
    content: Add run_training.sh docs page, update mkdocs nav, scripts index, training guide, and run-inference docs
    status: completed
isProject: false
---

# Local Run Scripts: Pre-Sync + Training Runner

## Changes

### 1. Add pre-sync to `run_inference.sh`

Before the SSH sbatch submission (around line 337), add a step that rsyncs local project data to the GPU via SSH:

```bash
# Pre-sync: push manual_data + project.json to GPU
echo "Syncing project data to GPU..."
rsync -az --progress -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/$PROJECT/manual_data/" "$SSH_DEST:$REMOTE_DIR/$PROJECT/manual_data/"
rsync -az -e "ssh $SSH_OPTS" \
    "$SCRIPT_DIR/$PROJECT/project.json" "$SSH_DEST:$REMOTE_DIR/$PROJECT/project.json"
```

Also add a `--no-push` flag to skip this pre-sync when not needed.

The sync uses direct SSH rsync (not SSHFS) for reliable upload. This goes right before the `scp` of the SLURM script at [run_inference.sh line 339-340](run_inference.sh).

### 2. Create `run_training.sh`

Same local-runner pattern as [run_inference.sh](run_inference.sh), but wrapping [submit_train.sh](submit_train.sh):

- **Same SSH boilerplate**: SSH config, ensure_ssh, remote helper, SSHFS mount check, cleanup trap, log streaming, squeue polling (reuse the same pattern from run_inference.sh lines 30-36, 322-332, 356-405)
- **Same args as submit_train.sh**: --project, --gpu, --epochs, --batch-size, --model, --label, --filter-classes, --sources, --manual-datasets, --infer-after, etc. (mirror [submit_train.sh](submit_train.sh) lines 116-146)
- **Same SLURM script generation**: Copy the heredoc template from submit_train.sh lines 267-517
- **Pre-sync**: Same `manual_data/` + `project.json` rsync as run_inference.sh (before sbatch)
- **Post-sync (lightweight only)**: After job completes, sync back only JSON metadata from the training run:

```bash
  RUN_DIR="$MOUNT_POINT/$OUTPUT_DIR"
  DST_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
  mkdir -p "$DST_DIR"
  # Only sync JSON files (class_info.json, results.json, training_config.json)
  rsync -a --include='*.json' --exclude='*' "$RUN_DIR/" "$DST_DIR/"


```

- **If `--infer-after`**: Also sync inference results (same as run_inference.sh post-sync)
- `**--no-sync` / `--no-push**`: Skip post-sync / pre-sync respectively

### 3. Update docs

- Add `run_training.sh` docs page at `docs/scripts/run-training.md`
- Add to [mkdocs.yml](mkdocs.yml) nav under SLURM Scripts
- Update [docs/scripts/index.md](docs/scripts/index.md) to list run_training.sh
- Update [docs/guides/training.md](docs/guides/training.md) to recommend local runner
- Update [docs/scripts/run-inference.md](docs/scripts/run-inference.md) to document `--no-push` and pre-sync behavior
