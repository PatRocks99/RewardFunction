# %%
"""Collect deploy-ready checkpoints from a completed training matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from training_notebook_common import FINAL_NAMES, ROOT


def read_rows(summary_csv: Path) -> list[dict[str, Any]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_run(row: dict[str, Any], destination: Path) -> dict[str, Any]:
    checkpoint = Path(row["final_path"])
    source_dir = checkpoint.parent
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, destination / checkpoint.name)
    for name in ["config.json", "offline_eval.json"]:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    metadata = {
        "reward_dataset": row["reward_dataset"],
        "split": row["split"],
        "lidar_beams": int(row["lidar_beams"]),
        "algorithm": row["algorithm"],
        "offline_action_mse": float(row["offline_action_mse"]),
        "offline_action_mae": float(row["offline_action_mae"]),
        "source_checkpoint": str(checkpoint),
        "copied_checkpoint": str(destination / checkpoint.name),
        "params": json.loads(row["params"]) if row.get("params") else {},
    }
    (destination / "selection_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def collect(summary_csv: Path, output_root: Path) -> list[dict[str, Any]]:
    rows = read_rows(summary_csv)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for row in rows:
        destination = (
            output_root
            / "by_algorithm"
            / row["reward_dataset"]
            / f"lidar_{row['lidar_beams']}"
            / row["split"]
            / row["algorithm"]
        )
        manifest.append(copy_run(row, destination))

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["reward_dataset"], row["lidar_beams"], row["split"])
        groups.setdefault(key, []).append(row)

    for (reward_dataset, lidar_beams, split), group_rows in groups.items():
        best = min(group_rows, key=lambda item: float(item["offline_action_mse"]))
        destination = output_root / "overall_best" / reward_dataset / f"lidar_{lidar_beams}" / split
        manifest.append(copy_run(best, destination))

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "reward_dataset",
            "split",
            "lidar_beams",
            "algorithm",
            "offline_action_mse",
            "offline_action_mae",
            "source_checkpoint",
            "copied_checkpoint",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in manifest:
            writer.writerow({key: item.get(key, "") for key in fieldnames})
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect best deploy-ready models.")
    parser.add_argument("--summary-csv", default=str(ROOT / "deploy_training_matrix" / "summary.csv"))
    parser.add_argument("--output-root", default=str(ROOT / "final_best_models"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manifest = collect(Path(args.summary_csv), Path(args.output_root))
    print(f"Copied {len(manifest)} model selection entries to {args.output_root}")
