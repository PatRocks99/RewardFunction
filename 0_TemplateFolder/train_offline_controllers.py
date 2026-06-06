from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONTROLLERS = ROOT / "Controllers"
if str(CONTROLLERS) not in sys.path:
    sys.path.insert(0, str(CONTROLLERS))

ALGORITHM_SPECS = {
    "bc": {
        "path": CONTROLLERS / "BC" / "train_bc.py",
        "module_name": "standalone_train_bc",
        "final": "final_bc.pt",
    },
    "awac": {
        "path": CONTROLLERS / "AWAC" / "train_awac.py",
        "module_name": "standalone_train_awac",
        "final": "final_awac.pt",
    },
    "iql": {
        "path": CONTROLLERS / "IQL" / "train_iql.py",
        "module_name": "standalone_train_iql",
        "final": "final_iql.pt",
    },
    "td3bc": {
        "path": CONTROLLERS / "TD3+BC" / "train_td3_bc.py",
        "module_name": "standalone_train_td3_bc",
        "final": "final_td3_bc.pt",
    },
}


RUN_ORDER = ["bc", "awac", "iql", "td3bc"]


def default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trainer_modules() -> dict[str, Any]:
    return {
        algorithm: load_module(spec["module_name"], spec["path"])
        for algorithm, spec in ALGORITHM_SPECS.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train BC, AWAC, IQL, and/or TD3+BC from a folder containing "
            ".h5, .hdf5, or .npz offline transition files."
        )
    )
    parser.add_argument("--data-folder", required=True, help="Folder containing dataset files.")
    parser.add_argument(
        "--algorithm",
        choices=[*RUN_ORDER, "all"],
        default="all",
        help="Controller to train. Default trains all four in sequence.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(ROOT / "standalone_runs"),
        help="Root folder for checkpoints. Each algorithm gets a subfolder.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint folder for a single-algorithm run. Ignored for --algorithm all.",
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional alternative to --max-steps. Steps are ceil(transitions / batch_size) * epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=None)

    # BC.
    parser.add_argument("--lr", type=float, default=None, help="BC learning rate.")

    # AWAC, IQL, and TD3+BC shared knobs.
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)

    # AWAC.
    parser.add_argument("--awac-lambda", dest="awac_lambda", type=float, default=None)
    parser.add_argument("--max-weight", type=float, default=None)

    # IQL.
    parser.add_argument("--vf-lr", type=float, default=None)
    parser.add_argument("--qf-lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--iql-tau", dest="iql_tau", type=float, default=None)
    parser.add_argument("--deterministic-actor", action="store_true", default=None)

    # TD3+BC.
    parser.add_argument("--policy-noise", type=float, default=None)
    parser.add_argument("--noise-clip", type=float, default=None)
    parser.add_argument("--policy-delay", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    return parser.parse_args()


def requested_algorithms(name: str) -> list[str]:
    return RUN_ORDER if name == "all" else [name]


def transition_count(data_folder: Path) -> int:
    from offline_common import load_h5_transitions

    dataset = load_h5_transitions(str(data_folder))
    return int(dataset["observations"].shape[0])


def supported_config_values(module: Any, args: argparse.Namespace, checkpoint_dir: Path) -> dict[str, Any]:
    supported = {field.name for field in fields(module.Config)}
    values: dict[str, Any] = {
        "data_folder": str(Path(args.data_folder).resolve()),
        "checkpoint_dir": str(checkpoint_dir),
        "device": args.device,
    }
    candidate_names = [
        "seed",
        "max_steps",
        "batch_size",
        "eval_freq",
        "hidden_dim",
        "hidden_layers",
        "normalize_obs",
        "normalize_actions",
        "lr",
        "actor_lr",
        "critic_lr",
        "discount",
        "tau",
        "awac_lambda",
        "max_weight",
        "vf_lr",
        "qf_lr",
        "beta",
        "iql_tau",
        "deterministic_actor",
        "policy_noise",
        "noise_clip",
        "policy_delay",
        "alpha",
    ]
    for name in candidate_names:
        value = getattr(args, name)
        if name in supported and value is not None:
            values[name] = value
    return values


def make_checkpoint_dir(args: argparse.Namespace, algorithm: str) -> Path:
    if args.algorithm != "all" and args.checkpoint_dir is not None:
        return Path(args.checkpoint_dir).resolve()
    data_name = Path(args.data_folder).resolve().name
    return Path(args.checkpoint_root).resolve() / data_name / algorithm


def apply_epoch_steps(config: Any, epochs: int, num_transitions: int) -> Any:
    steps = int(math.ceil(num_transitions / float(config.batch_size)) * epochs)
    config.max_steps = steps
    return config


def train_algorithm(algorithm: str, module: Any, args: argparse.Namespace, num_transitions: int | None) -> Path:
    checkpoint_dir = make_checkpoint_dir(args, algorithm)
    config_values = supported_config_values(module, args, checkpoint_dir)
    config = module.Config(**config_values)
    if args.epochs is not None:
        if num_transitions is None:
            num_transitions = transition_count(Path(args.data_folder).resolve())
        config = apply_epoch_steps(config, args.epochs, num_transitions)

    print(f"\n=== Training {algorithm.upper()} ===")
    print(json.dumps(asdict(config), indent=2))
    final_path = module.train(config)
    print(f"Saved final {algorithm.upper()} model to {final_path}")
    return final_path


def main() -> int:
    args = parse_args()
    data_folder = Path(args.data_folder).resolve()
    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")
    if not data_folder.is_dir():
        raise NotADirectoryError(f"--data-folder must be a folder: {data_folder}")

    algorithms = requested_algorithms(args.algorithm)
    modules = trainer_modules()
    num_transitions = transition_count(data_folder) if args.epochs is not None else None

    final_paths: dict[str, str] = {}
    for algorithm in algorithms:
        final_path = train_algorithm(algorithm, modules[algorithm], args, num_transitions)
        final_paths[algorithm] = str(final_path)

    print("\n=== Finished ===")
    print(json.dumps(final_paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
