# %%
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from offline_common import (  # noqa: E402
    DeterministicActor,
    ReplayBuffer,
    load_h5_transitions,
    normalize_dataset,
    prepare_checkpoint_dir,
    save_checkpoint,
    set_seed,
)


# %%
@dataclass
class Config:
    data_folder: str
    checkpoint_dir: str = "runs/bc"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    max_steps: int = 50000
    batch_size: int = 1024
    lr: float = 3e-4
    eval_freq: int = 5000
    normalize_obs: bool = True
    normalize_actions: bool = False
    hidden_dim: int = 256
    hidden_layers: int = 2


# %%
def train(config: Config) -> Path:
    set_seed(config.seed)
    checkpoint_dir = prepare_checkpoint_dir(config.checkpoint_dir, config)

    dataset = load_h5_transitions(config.data_folder)
    stats = normalize_dataset(dataset, config.normalize_obs, config.normalize_actions)
    buffer = ReplayBuffer(dataset, config.device)

    actor = DeterministicActor(
        buffer.obs_dim,
        buffer.action_dim,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    ).to(config.device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=config.lr)

    best_loss = float("inf")
    for step in range(1, config.max_steps + 1):
        observations, actions, *_ = buffer.sample(config.batch_size)
        predictions = actor(observations)
        loss = F.mse_loss(predictions, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.eval_freq == 0 or step == 1:
            print(f"step={step} bc_loss={loss.item():.6f}")
            payload = {
                "actor": actor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "loss": loss.item(),
                "stats": stats,
            }
            save_checkpoint(checkpoint_dir / f"checkpoint_{step}.pt", payload)
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_checkpoint(checkpoint_dir / "best_bc.pt", payload)

    final_path = checkpoint_dir / "final_bc.pt"
    save_checkpoint(final_path, {"actor": actor.state_dict(), "step": config.max_steps, "stats": stats})
    return final_path


# %%
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy on offline transition files.")
    parser.add_argument("--data-folder", required=True, help="Folder containing .h5/.hdf5/.npz training files.")
    parser.add_argument("--checkpoint-dir", default="runs/bc", help="Folder where checkpoints will be saved.")
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    return Config(**vars(parser.parse_args()))


# %%
if __name__ == "__main__":
    saved_path = train(parse_args())
    print(f"Saved final model to {saved_path}")
