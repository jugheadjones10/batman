#!/usr/bin/env python3
"""
Resume experiments from a multirun directory where training completed but inference didn't.

This script is useful when RF-DETR's sys.exit() prevented the post-training steps
(inference and summary generation) from running.

Usage:
    # Resume all experiments via SLURM (recommended)
    python experiments/resume_experiments.py --slurm experiments/multirun/2026-01-30_17-14-22/

    # Resume locally (only for testing, not recommended)
    python experiments/resume_experiments.py --local experiments/multirun/2026-01-30_17-14-22/

    # Resume a specific experiment via SLURM
    python experiments/resume_experiments.py --slurm experiments/multirun/2026-01-30_17-14-22/exp_person_25

    # Dry run (show what would be done)
    python experiments/resume_experiments.py --slurm --dry-run experiments/multirun/2026-01-30_17-14-22/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# SLURM configuration (NUS SoC Compute Cluster)
SLURM_CONFIG = {
    "partition": "gpu-long",
    "gpu_type": "h100-96",
    "timeout_min": 120,  # 2 hours should be enough for inference only
    "cpus_per_task": 8,
    "mem_gb": 64,
}


def find_experiments_to_resume(base_dir: Path) -> list[tuple[Path, str]]:
    """
    Find experiments that need to be resumed.
    
    An experiment needs resuming if:
    - It has a training/checkpoint*.pth file (training completed)
    - It doesn't have experiment_summary.json (post-training didn't complete)
    
    Returns:
        List of (experiment_dir, experiment_name) tuples
    """
    experiments_to_resume = []
    
    # Check if base_dir is a single experiment or a multirun directory
    if (base_dir / "training").exists():
        # Single experiment directory
        exp_name = base_dir.name
        training_dir = base_dir / "training"
        summary_file = base_dir / "experiment_summary.json"
        
        has_checkpoint = any(training_dir.glob("checkpoint*.pth"))
        has_summary = summary_file.exists()
        
        if has_checkpoint and not has_summary:
            experiments_to_resume.append((base_dir, exp_name))
    else:
        # Multirun directory - check each subdirectory
        for exp_dir in sorted(base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            
            training_dir = exp_dir / "training"
            summary_file = exp_dir / "experiment_summary.json"
            
            if not training_dir.exists():
                continue
            
            has_checkpoint = any(training_dir.glob("checkpoint*.pth"))
            has_summary = summary_file.exists()
            
            if has_checkpoint and not has_summary:
                experiments_to_resume.append((exp_dir, exp_dir.name))
    
    return experiments_to_resume


def resume_experiment_local(exp_dir: Path, exp_name: str, dry_run: bool = False) -> bool:
    """
    Resume a single experiment locally.
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Resuming (LOCAL): {exp_name}")
    print(f"Directory: {exp_dir}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "experiments/train_experiment.py",
        f"experiment={exp_name}",
        f"resume_from={exp_dir}",
    ]
    
    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to resume {exp_name}: {e}")
        return False


def resume_experiment_slurm(exp_dir: Path, exp_name: str, dry_run: bool = False) -> bool:
    """
    Resume a single experiment via SLURM.
    
    Returns:
        True if job submitted successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Submitting to SLURM: {exp_name}")
    print(f"Directory: {exp_dir}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable,
        "experiments/train_experiment.py",
        f"experiment={exp_name}",
        f"resume_from={exp_dir}",
        "hydra/launcher=submitit_slurm",
        f"hydra.launcher.partition={SLURM_CONFIG['partition']}",
        f"hydra.launcher.timeout_min={SLURM_CONFIG['timeout_min']}",
        f"hydra.launcher.cpus_per_task={SLURM_CONFIG['cpus_per_task']}",
        f"hydra.launcher.mem_gb={SLURM_CONFIG['mem_gb']}",
        f"hydra.launcher.gres=gpu:{SLURM_CONFIG['gpu_type']}:1",
    ]
    
    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            check=True,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to submit {exp_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Resume experiments from a multirun directory"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to multirun directory or single experiment directory",
    )
    
    # Execution mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--slurm",
        action="store_true",
        help="Submit jobs to SLURM (recommended)",
    )
    mode_group.add_argument(
        "--local",
        action="store_true",
        help="Run locally (not recommended for heavy workloads)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running",
    )
    
    args = parser.parse_args()
    
    if not args.directory.exists():
        print(f"ERROR: Directory not found: {args.directory}")
        sys.exit(1)
    
    # Find experiments to resume
    experiments = find_experiments_to_resume(args.directory)
    
    if not experiments:
        print(f"No experiments need resuming in {args.directory}")
        print("\nPossible reasons:")
        print("  - All experiments already have experiment_summary.json")
        print("  - No training checkpoints found")
        sys.exit(0)
    
    print(f"Found {len(experiments)} experiment(s) to resume:")
    for exp_dir, exp_name in experiments:
        print(f"  - {exp_name}: {exp_dir}")
    
    mode_str = "SLURM" if args.slurm else "LOCAL"
    print(f"\nExecution mode: {mode_str}")
    
    if args.dry_run:
        print("[DRY RUN] Will show commands without executing")
    
    # Resume each experiment
    results = []
    for exp_dir, exp_name in experiments:
        if args.slurm:
            success = resume_experiment_slurm(exp_dir, exp_name, args.dry_run)
        else:
            success = resume_experiment_local(exp_dir, exp_name, args.dry_run)
        results.append((exp_name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    succeeded = sum(1 for _, success in results if success)
    failed = len(results) - succeeded
    
    for exp_name, success in results:
        status = "SUBMITTED" if (success and args.slurm) else ("OK" if success else "FAILED")
        print(f"  {exp_name}: {status}")
    
    print(f"\nTotal: {succeeded} succeeded, {failed} failed")
    
    if args.slurm and not args.dry_run:
        print("\nMonitor jobs with: squeue -u $USER")
        print("After completion, collect results with:")
        print("  python experiments/collect_results.py --latest")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
