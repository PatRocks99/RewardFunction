# %%
import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).resolve().parents[1]))
from offline_common import (  # noqa: E402
    GaussianActor,
    ReplayBuffer,
    TwinQ,
    load_h5_transitions,
    normalize_dataset,
    prepare_checkpoint_dir,
    save_checkpoint,
    set_seed,
    soft_update,
)


# %%
@dataclass
class Config:
    data_folder: str
    checkpoint_dir: str = "runs/awac"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    max_steps: int = 100000
    batch_size: int = 256
    eval_freq: int = 5000
    discount: float = 0.99
    tau: float = 0.005
    awac_lambda: float = 1.0
    max_weight: float = 20.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
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

    actor = GaussianActor(buffer.obs_dim, buffer.action_dim, config.hidden_dim, config.hidden_layers).to(config.device)
    critic = TwinQ(buffer.obs_dim, buffer.action_dim, config.hidden_dim, config.hidden_layers).to(config.device)
    critic_target = copy.deepcopy(critic).requires_grad_(False).to(config.device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    best_actor_loss = float("inf")

    for step in range(1, config.max_steps + 1):
        observations, actions, rewards, next_observations, terminals = buffer.sample(config.batch_size)

        with torch.no_grad():
            next_actions, _ = actor.sample(next_observations)
            target_q = critic_target(next_observations, next_actions)
            target = rewards + (1.0 - terminals) * config.discount * target_q

        q1, q2 = critic.both(observations, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        soft_update(critic_target, critic, config.tau)

        with torch.no_grad():
            sampled_actions, _ = actor.sample(observations)
            v = critic(observations, sampled_actions)
            q = critic(observations, actions)
            weights = torch.exp((q - v) / config.awac_lambda).clamp(max=config.max_weight)

        dist = actor(observations)
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        actor_loss = -(weights * log_prob).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        if step % config.eval_freq == 0 or step == 1:
            print(f"step={step} critic_loss={critic_loss.item():.6f} actor_loss={actor_loss.item():.6f}")
            payload = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "step": step,
                "stats": stats,
            }
            save_checkpoint(checkpoint_dir / f"checkpoint_{step}.pt", payload)
            if actor_loss.item() < best_actor_loss:
                best_actor_loss = actor_loss.item()
                save_checkpoint(checkpoint_dir / "best_awac.pt", payload)

    final_path = checkpoint_dir / "final_awac.pt"
    save_checkpoint(final_path, {"actor": actor.state_dict(), "critic": critic.state_dict(), "stats": stats})
    return final_path


# %%
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train AWAC on offline transition files.")
    parser.add_argument("--data-folder", required=True, help="Folder containing .h5/.hdf5/.npz training files.")
    parser.add_argument("--checkpoint-dir", default="runs/awac", help="Folder where checkpoints will be saved.")
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--awac-lambda", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=20.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    return Config(**vars(parser.parse_args()))


# %%
if __name__ == "__main__":
    saved_path = train(parse_args())
    print(f"Saved final model to {saved_path}")
