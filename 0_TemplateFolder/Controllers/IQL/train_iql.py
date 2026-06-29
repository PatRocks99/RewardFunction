# %%
import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(str(Path(__file__).resolve().parents[1]))
from offline_common import (  # noqa: E402
    DeterministicActor,
    GaussianActor,
    ReplayBuffer,
    TwinQ,
    ValueFunction,
    load_h5_transitions,
    normalize_dataset,
    prepare_checkpoint_dir,
    save_checkpoint,
    set_seed,
    soft_update,
)


# %%
EXP_ADV_MAX = 100.0


@dataclass
class Config:
    data_folder: str
    checkpoint_dir: str = "runs/iql"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    max_steps: int = 100000
    batch_size: int = 256
    eval_freq: int = 5000
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    deterministic_actor: bool = False
    normalize_obs: bool = True
    normalize_actions: bool = False
    hidden_dim: int = 256
    hidden_layers: int = 2


# %%
def asymmetric_l2_loss(error: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.abs(tau - (error < 0).float())
    return (weight * error.pow(2)).mean()


def actor_bc_loss(actor, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "log_prob"):
        return -actor.log_prob(observations, actions)
    output = actor(observations)
    if isinstance(output, torch.distributions.Distribution):
        return -output.log_prob(actions).sum(dim=-1, keepdim=True)
    return (output - actions).pow(2).sum(dim=-1, keepdim=True)


# %%
def train(config: Config) -> Path:
    set_seed(config.seed)
    checkpoint_dir = prepare_checkpoint_dir(config.checkpoint_dir, config)

    dataset = load_h5_transitions(config.data_folder)
    stats = normalize_dataset(dataset, config.normalize_obs, config.normalize_actions)
    buffer = ReplayBuffer(dataset, config.device)

    qf = TwinQ(buffer.obs_dim, buffer.action_dim, config.hidden_dim, config.hidden_layers).to(config.device)
    q_target = copy.deepcopy(qf).requires_grad_(False).to(config.device)
    vf = ValueFunction(buffer.obs_dim, config.hidden_dim, config.hidden_layers).to(config.device)
    actor_cls = DeterministicActor if config.deterministic_actor else GaussianActor
    actor = actor_cls(buffer.obs_dim, buffer.action_dim, config.hidden_dim, config.hidden_layers).to(config.device)

    q_optimizer = torch.optim.Adam(qf.parameters(), lr=config.qf_lr)
    v_optimizer = torch.optim.Adam(vf.parameters(), lr=config.vf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    actor_scheduler = CosineAnnealingLR(actor_optimizer, config.max_steps)

    for step in range(1, config.max_steps + 1):
        observations, actions, rewards, next_observations, terminals = buffer.sample(config.batch_size)

        with torch.no_grad():
            target_q = q_target(observations, actions)
        v = vf(observations)
        advantage = target_q - v
        v_loss = asymmetric_l2_loss(advantage, config.iql_tau)
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

        with torch.no_grad():
            next_v = vf(next_observations)
            q_target_value = rewards + (1.0 - terminals) * config.discount * next_v
        q1, q2 = qf.both(observations, actions)
        q_loss = F.mse_loss(q1, q_target_value) + F.mse_loss(q2, q_target_value)
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()
        soft_update(q_target, qf, config.tau)

        weights = torch.exp(config.beta * advantage.detach()).clamp(max=EXP_ADV_MAX)
        policy_loss = (weights * actor_bc_loss(actor, observations, actions)).mean()
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
        actor_scheduler.step()

        if step % config.eval_freq == 0 or step == 1:
            print(
                f"step={step} value_loss={v_loss.item():.6f} "
                f"q_loss={q_loss.item():.6f} actor_loss={policy_loss.item():.6f}"
            )
            save_checkpoint(
                checkpoint_dir / f"checkpoint_{step}.pt",
                {
                    "actor": actor.state_dict(),
                    "qf": qf.state_dict(),
                    "vf": vf.state_dict(),
                    "q_optimizer": q_optimizer.state_dict(),
                    "v_optimizer": v_optimizer.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "actor_scheduler": actor_scheduler.state_dict(),
                    "step": step,
                    "stats": stats,
                },
            )

    final_path = checkpoint_dir / "final_iql.pt"
    save_checkpoint(final_path, {"actor": actor.state_dict(), "qf": qf.state_dict(), "vf": vf.state_dict(), "stats": stats})
    return final_path


# %%
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train IQL on offline transition files.")
    parser.add_argument("--data-folder", required=True, help="Folder containing .h5/.hdf5/.npz training files.")
    parser.add_argument("--checkpoint-dir", default="runs/iql", help="Folder where checkpoints will be saved.")
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--iql-tau", type=float, default=0.7)
    parser.add_argument("--vf-lr", type=float, default=3e-4)
    parser.add_argument("--qf-lr", type=float, default=3e-4)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--deterministic-actor", action="store_true")
    parser.add_argument("--normalize-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=2)
    return Config(**vars(parser.parse_args()))


# %%
if __name__ == "__main__":
    saved_path = train(parse_args())
    print(f"Saved final model to {saved_path}")
