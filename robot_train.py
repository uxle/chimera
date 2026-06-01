"""
ChimeraRobot Embodied Training
--------------------------------
Three training modes for the robot brain:

  1. Imitation Learning (IL)
     Record human demonstrations → train on them
     The robot copies what a human does (teleop recordings)
     Fast to train, needs human data

  2. Reinforcement Learning (RL) — PPO
     Robot explores the simulator, gets reward signals
     Learns entirely from self-play and trial/error
     Slow to train but can discover novel strategies

  3. Joint Training (IL + RL)
     Pre-train on demonstrations, then refine with RL
     Best of both worlds — recommended

  4. Language Grounding
     Train the language model to describe what the robot sees/feels
     Uses pairs of (sensory frame, text description)

Run:
    python robot_train.py --mode il    --demos demos/
    python robot_train.py --mode rl    --steps 100000
    python robot_train.py --mode joint --demos demos/ --steps 50000
"""

import argparse
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from robot_config import RobotConfig, BrainConfig
from robot_brain import RobotBrain, BrainOutput
from simulator import RobotSimulator
from multimodal import MultimodalFusion, SensoryFrame
from emotion import EmotionSystem


# ─────────────────────────────────────────────────────────────────────
#  Demonstration dataset (for imitation learning)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Demonstration:
    """One human-recorded robot control demonstration."""
    obs:        dict        # Observation (raw sensors)
    action:     dict        # Action taken by human operator
    reward:     float = 0.0
    done:       bool  = False


class DemonstrationDataset(Dataset):
    """
    Dataset of human demonstrations for imitation learning.
    Each sample: (observation features, action)
    """

    def __init__(self, demos_dir: str, fusion: MultimodalFusion):
        self.demos: List[Demonstration] = []
        self.fusion = fusion
        self._load(demos_dir)

    def _load(self, demos_dir: str):
        if not os.path.isdir(demos_dir):
            print(f"[IL] No demo dir found: {demos_dir}")
            return
        for fn in sorted(os.listdir(demos_dir)):
            if not fn.endswith(".jsonl"):
                continue
            with open(os.path.join(demos_dir, fn)) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self.demos.append(Demonstration(**d))
                    except Exception:
                        pass
        print(f"[IL] Loaded {len(self.demos)} demonstrations")

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        demo   = self.demos[idx]
        action = demo.action
        # Build a sensory frame from obs (simplified — use noise as placeholders)
        frame = SensoryFrame(
            vision_emb=torch.randn(512),
            audio_emb=torch.randn(256),
            touch_emb=torch.randn(64),
            proprio_emb=torch.randn(128),
        )
        with torch.no_grad():
            obs_emb = self.fusion(frame)

        # Action vector: normalise to [-1, 1]
        act_vec = torch.tensor([
            float(action.get("head_pan",  0.0)),
            float(action.get("head_tilt", 0.0)),
            float(action.get("speed",     0.0)) * 2 - 1,
            float(action.get("force",     0.0)) * 2 - 1,
        ], dtype=torch.float32)
        # Pad to n_action_dims
        n = 32
        if act_vec.shape[0] < n:
            act_vec = F.pad(act_vec, (0, n - act_vec.shape[0]))

        return obs_emb.detach(), act_vec


# ─────────────────────────────────────────────────────────────────────
#  PPO Actor-Critic (for RL)
# ─────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    PPO Actor: maps world embedding → action distribution.
    Uses a Gaussian policy for continuous actions.
    """

    def __init__(self, d_model: int = 512, n_actions: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )
        self.mu_head  = nn.Linear(256, n_actions)
        self.log_std  = nn.Parameter(torch.zeros(n_actions))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h  = self.body(obs)
        mu = torch.tanh(self.mu_head(h))
        return mu, self.log_std.exp()

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self(obs)
        dist    = torch.distributions.Normal(mu, std)
        action  = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mu, std = self(obs)
        dist    = torch.distributions.Normal(mu, std)
        return dist.log_prob(action).sum(-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        mu, std = self(obs)
        dist    = torch.distributions.Normal(mu, std)
        return dist.entropy().sum(-1)


class ValueNetwork(nn.Module):
    """PPO Critic: maps world embedding → scalar value estimate."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 512), nn.Tanh(),
            nn.Linear(512, 256),     nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────
#  Experience buffer for PPO
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RLExperience:
    obs:      torch.Tensor
    action:   torch.Tensor
    reward:   float
    done:     bool
    log_prob: float
    value:    float


class RolloutBuffer:
    def __init__(self, capacity: int = 2048):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.obs:      List[torch.Tensor] = []
        self.actions:  List[torch.Tensor] = []
        self.rewards:  List[float]        = []
        self.dones:    List[bool]         = []
        self.log_probs:List[float]        = []
        self.values:   List[float]        = []

    def add(self, exp: RLExperience):
        self.obs.append(exp.obs)
        self.actions.append(exp.action)
        self.rewards.append(exp.reward)
        self.dones.append(exp.done)
        self.log_probs.append(exp.log_prob)
        self.values.append(exp.value)

    def __len__(self): return len(self.rewards)

    def is_full(self): return len(self) >= self.capacity

    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95) -> torch.Tensor:
        """Compute GAE (Generalised Advantage Estimation) returns."""
        T = len(self.rewards)
        advantages = torch.zeros(T)
        gae = 0.0
        for t in reversed(range(T)):
            next_val = self.values[t+1] if t < T-1 else 0.0
            delta    = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae      = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
        returns = advantages + torch.tensor(self.values)
        return returns, advantages


# ─────────────────────────────────────────────────────────────────────
#  Imitation learner
# ─────────────────────────────────────────────────────────────────────

class ImitationLearner:
    """Trains action decoder via behaviour cloning on demonstrations."""

    def __init__(self, fusion: MultimodalFusion, n_actions: int = 32,
                 lr: float = 1e-4, device: str = "cpu"):
        self.policy   = PolicyNetwork(fusion.fusion.out_proj[0].out_features
                                       if hasattr(fusion, 'fusion') else 512, n_actions)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        self.device    = device
        self.policy.to(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.policy.train()
        total_loss = 0.0
        n = 0
        for obs, target_action in dataloader:
            obs           = obs.to(self.device)
            target_action = target_action.to(self.device)
            self.optimizer.zero_grad()
            mu, std = self.policy(obs)
            # Behaviour cloning loss: MSE between predicted mu and target
            loss = F.mse_loss(mu, target_action)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────
#  PPO trainer
# ─────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Proximal Policy Optimisation for robot embodied RL.
    Trains in the RobotSimulator environment.
    """

    def __init__(
        self,
        policy:      PolicyNetwork,
        value_net:   ValueNetwork,
        d_model:     int   = 512,
        lr:          float = 3e-4,
        clip_eps:    float = 0.2,
        n_epochs:    int   = 10,
        batch_size:  int   = 256,
        gamma:       float = 0.99,
        lam:         float = 0.95,
        ent_coef:    float = 0.01,
        device:      str   = "cpu",
    ):
        self.policy    = policy.to(device)
        self.value_net = value_net.to(device)
        self.clip_eps  = clip_eps
        self.n_epochs  = n_epochs
        self.batch_size = batch_size
        self.gamma     = gamma
        self.lam       = lam
        self.ent_coef  = ent_coef
        self.device    = device
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()), lr=lr
        )
        self.buffer = RolloutBuffer(capacity=2048)

    def update(self) -> Dict[str, float]:
        """Run PPO update on collected rollout."""
        returns, advantages = self.buffer.compute_returns(self.gamma, self.lam)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs     = torch.stack(self.buffer.obs).to(self.device)
        actions = torch.stack(self.buffer.actions).to(self.device)
        old_lps = torch.tensor(self.buffer.log_probs, device=self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(self.n_epochs):
            idx = torch.randperm(obs.shape[0])
            for start in range(0, obs.shape[0], self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                b_obs   = obs[batch_idx]
                b_act   = actions[batch_idx]
                b_old   = old_lps[batch_idx]
                b_ret   = returns[batch_idx]
                b_adv   = advantages[batch_idx]

                new_lp  = self.policy.log_prob(b_obs, b_act)
                ratio   = (new_lp - b_old).exp()
                surr1   = ratio * b_adv
                surr2   = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                p_loss  = -torch.min(surr1, surr2).mean()

                v_pred  = self.value_net(b_obs)
                v_loss  = F.mse_loss(v_pred, b_ret)

                entropy = self.policy.entropy(b_obs).mean()
                loss    = p_loss + 0.5 * v_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()), 0.5
                )
                self.optimizer.step()

                metrics["policy_loss"] += p_loss.item()
                metrics["value_loss"]  += v_loss.item()
                metrics["entropy"]     += entropy.item()

        self.buffer.clear()
        return {k: v / max(1, self.n_epochs) for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────
#  Embodied training loop
# ─────────────────────────────────────────────────────────────────────

class EmbodiedTrainer:
    """Master training loop combining IL and RL."""

    def __init__(self, config: RobotConfig, device: str = "cpu"):
        self.config = config
        self.device = device
        self.sim    = RobotSimulator(config)
        self.fusion = MultimodalFusion(config.brain).to(device)
        self.policy = PolicyNetwork(config.brain.d_model, config.brain.n_action_dims).to(device)
        self.value  = ValueNetwork(config.brain.d_model).to(device)
        self.ppo    = PPOTrainer(self.policy, self.value, device=device)
        self._step  = 0
        self._ep_rewards: Deque[float] = deque(maxlen=50)
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _obs_to_embedding(self, obs: dict) -> torch.Tensor:
        """Convert raw simulator obs dict to fused embedding."""
        frame = SensoryFrame(
            vision_emb=torch.randn(self.config.brain.vision_feature_dim),
            audio_emb=torch.randn(self.config.brain.audio_feature_dim),
            touch_emb=torch.tensor([obs.get("touch_right", 0.0)] * self.config.brain.touch_feature_dim),
            proprio_emb=torch.randn(self.config.brain.proprio_feature_dim),
            is_speech=bool(obs.get("hearing_speech", False)),
        )
        with torch.no_grad():
            return self.fusion(frame).to(self.device)

    def train_rl(self, total_steps: int = 100000):
        """Train with PPO in the simulator."""
        print(f"\n[RL] Training for {total_steps} steps...")
        obs     = self.sim.reset()
        ep_rew  = 0.0

        while self._step < total_steps:
            emb = self._obs_to_embedding(obs)
            with torch.no_grad():
                action, log_prob = self.policy.sample(emb.unsqueeze(0))
                value = self.value(emb.unsqueeze(0))

            action_dict = self._tensor_to_action(action.squeeze(0))
            reward, done, info = self.sim.apply_action(action_dict)
            ep_rew += reward

            self.ppo.buffer.add(RLExperience(
                obs=emb.detach().cpu(),
                action=action.squeeze(0).detach().cpu(),
                reward=reward,
                done=done,
                log_prob=float(log_prob),
                value=float(value),
            ))

            if done:
                self._ep_rewards.append(ep_rew)
                ep_rew = 0.0
                obs = self.sim.reset()
            else:
                obs, _, _, _ = self.sim.observe(), None, None, None
                obs = self.sim.observe()

            self._step += 1

            if self.ppo.buffer.is_full():
                metrics = self.ppo.update()
                avg_rew = sum(self._ep_rewards) / max(len(self._ep_rewards), 1)
                print(f"  Step {self._step:7d} | avg_rew={avg_rew:6.2f} | "
                      f"p_loss={metrics['policy_loss']:.4f} | "
                      f"v_loss={metrics['value_loss']:.4f}")

            if self._step % 10000 == 0:
                self._save(f"rl_step_{self._step}")

        print("[RL] Training complete")

    def train_il(self, demos_dir: str, epochs: int = 20, batch_size: int = 64):
        """Pre-train with imitation learning."""
        from torch.utils.data import DataLoader
        dataset = DemonstrationDataset(demos_dir, self.fusion)
        if len(dataset) == 0:
            print("[IL] No demonstrations found — skipping IL phase")
            return
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        il = ImitationLearner(self.fusion, self.config.brain.n_action_dims,
                              device=self.device)
        print(f"\n[IL] Training for {epochs} epochs on {len(dataset)} demos...")
        for ep in range(epochs):
            loss = il.train_epoch(loader)
            print(f"  Epoch {ep+1}/{epochs} | loss={loss:.4f}")
        # Copy IL policy weights to RL policy
        self.policy.load_state_dict(il.policy.state_dict(), strict=False)
        self._save("il_pretrained")
        print("[IL] Complete — weights transferred to RL policy")

    def _tensor_to_action(self, action: torch.Tensor) -> dict:
        """Convert action tensor to simulator-readable dict."""
        a = action.tolist()
        from motor import GaitMode, HandGesture
        gaits = list(GaitMode)
        gestures = list(HandGesture)
        return {
            "gait":       gaits[int(abs(a[0]) * (len(gaits)-1)) % len(gaits)].name,
            "gesture":    gestures[int(abs(a[1]) * (len(gestures)-1)) % len(gestures)].name,
            "speed":      max(0, min(1, (a[2] + 1) / 2)),
            "force":      max(0, min(1, (a[3] + 1) / 2)),
            "head_pan":   a[4] if len(a) > 4 else 0.0,
            "head_tilt":  a[5] if len(a) > 5 else 0.0,
        }

    def _save(self, name: str):
        path = os.path.join(self.config.checkpoint_dir, f"{name}.pt")
        torch.save({
            "policy":    self.policy.state_dict(),
            "value":     self.value.state_dict(),
            "fusion":    self.fusion.state_dict(),
            "step":      self._step,
        }, path)
        print(f"  [Save] {path}")


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",   choices=["il","rl","joint"], default="rl")
    p.add_argument("--demos",  default="demos/")
    p.add_argument("--steps",  type=int, default=100000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu")
    if args.device != "auto":
        device = args.device

    config  = RobotConfig(simulation_mode=True)
    trainer = EmbodiedTrainer(config, device=device)

    if args.mode in ("il", "joint"):
        trainer.train_il(args.demos, epochs=args.epochs)
    if args.mode in ("rl", "joint"):
        trainer.train_rl(total_steps=args.steps)


if __name__ == "__main__":
    main()
