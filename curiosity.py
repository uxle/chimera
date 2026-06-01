"""
ChimeraRobot Curiosity & Intrinsic Motivation
----------------------------------------------
A robot without curiosity just does what it's told.
A robot WITH curiosity actively explores, learns, and grows.

This module implements curiosity as an intrinsic reward signal:
  Curiosity reward = how surprised was the robot by what happened?

Architecture:
  Forward Model  — predicts the next sensory state given current state + action
  Inverse Model  — predicts what action was taken given state transitions
  Novelty Memory — tracks which states have been seen before
  Curiosity ICM  — Intrinsic Curiosity Module (Pathak et al. 2017)

The robot is rewarded for visiting states it can't yet predict.
As it learns, familiar things stop being interesting.
New things → high curiosity → exploration.

This is the closest artificial equivalent to how a baby's brain
rewards itself for exploring new objects and environments.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CuriositySignal:
    intrinsic_reward: float   # How curious/surprised (0–1)
    prediction_error: float   # How wrong our world model was
    novelty_score:    float   # How new is this state?
    recommended_action: str   # "explore" | "exploit" | "approach" | "avoid"
    interest_direction: Optional[np.ndarray] = None  # Vector toward interesting thing


# ─────────────────────────────────────────────────────────────────────
#  Forward model (world model)
# ─────────────────────────────────────────────────────────────────────

class ForwardModel(nn.Module):
    """
    Predicts what the world will look like after taking an action.
    If the prediction is very wrong → the robot was surprised → high curiosity.
    If the prediction is accurate → familiar, less interesting.

    Input:  current_state (d_model) + action (n_action_dims)
    Output: predicted_next_state (d_model)
    """

    def __init__(self, d_model: int = 512, n_action_dims: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + n_action_dims, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, d_model),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (B, d_model)
            action: (B, n_action_dims)
        Returns: (B, d_model) predicted next state
        """
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

    def prediction_error(
        self,
        state:      torch.Tensor,
        action:     torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prediction error (surprise).
        Higher error = more curious about this state.
        """
        predicted = self(state, action)
        return F.mse_loss(predicted, next_state.detach(), reduction="none").mean(-1)


# ─────────────────────────────────────────────────────────────────────
#  Inverse model (self-model)
# ─────────────────────────────────────────────────────────────────────

class InverseModel(nn.Module):
    """
    Predicts what action was taken given a state transition (s → s').
    Trains a self-model: "what did I do to cause this change?"
    Used to filter out environment noise from agent-caused changes.

    Input:  current_state + next_state
    Output: predicted action
    """

    def __init__(self, d_model: int = 512, n_action_dims: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.SiLU(),
            nn.Linear(256, n_action_dims),
        )

    def forward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────
#  Novelty memory (count-based exploration)
# ─────────────────────────────────────────────────────────────────────

class NoveltyMemory:
    """
    Tracks which embedding regions have been visited.
    Uses random projection to hash high-dimensional states to buckets.
    Unvisited regions have high novelty → high curiosity reward.

    This is "count-based exploration" — a fundamental technique in RL.
    It prevents the robot from getting stuck re-exploring the same things.
    """

    def __init__(self, d_model: int = 512, n_buckets: int = 4096, decay: float = 0.9999):
        self.n_buckets = n_buckets
        self.decay     = decay
        # Random projection matrix (fixed — not learned)
        self._proj = nn.Linear(d_model, int(math.log2(n_buckets)), bias=False)
        nn.init.normal_(self._proj.weight)
        for p in self._proj.parameters():
            p.requires_grad = False
        # Visit count per bucket
        self._counts = np.zeros(n_buckets, dtype=np.float32)
        self._total_visits = 0

    def _hash(self, embedding: torch.Tensor) -> int:
        """Project embedding to bucket index."""
        with torch.no_grad():
            proj = self._proj(embedding.unsqueeze(0)).squeeze(0)
            bits = (proj > 0).int().tolist()
            idx  = sum(b * (2 ** i) for i, b in enumerate(bits))
            return idx % self.n_buckets

    def novelty(self, embedding: torch.Tensor) -> float:
        """
        Novelty score in [0, 1].
        1.0 = never visited this region before
        0.0 = visited very many times
        """
        bucket = self._hash(embedding)
        count  = self._counts[bucket]
        # Novelty inversely proportional to sqrt(visit count)
        novelty = 1.0 / (1.0 + math.sqrt(count))
        return round(novelty, 4)

    def record_visit(self, embedding: torch.Tensor):
        """Mark this state as visited."""
        bucket = self._hash(embedding)
        self._counts[bucket] += 1.0
        self._total_visits   += 1
        # Decay all counts slowly (so old visits matter less)
        if self._total_visits % 100 == 0:
            self._counts *= self.decay

    def most_novel_direction(
        self,
        candidates: List[torch.Tensor],
    ) -> int:
        """Given a list of candidate state embeddings, return index of most novel."""
        scores = [self.novelty(c) for c in candidates]
        return int(np.argmax(scores))

    def exploration_stats(self) -> dict:
        visited = int((self._counts > 0).sum())
        return {
            "buckets_visited": visited,
            "total_buckets":   self.n_buckets,
            "coverage_%":      round(100 * visited / self.n_buckets, 1),
            "total_visits":    self._total_visits,
        }


# ─────────────────────────────────────────────────────────────────────
#  Curiosity drive (combines all signals)
# ─────────────────────────────────────────────────────────────────────

class CuriosityDrive:
    """
    Combines forward model, inverse model, and novelty memory
    into a unified curiosity signal.

    The curiosity reward is used to:
      1. Bias action selection toward novel/surprising states
      2. Amplify learning rate when surprised
      3. Generate "explore" vs "exploit" decisions
      4. Guide attention toward interesting parts of the scene

    Key property: curiosity decreases as the robot learns.
    A robot that has seen everything will stop being curious.
    A robot in a new environment will be very curious.
    """

    EXPLORE_THRESHOLD = 0.5    # Novelty above this → recommend exploring
    EXPLOIT_THRESHOLD = 0.2    # Novelty below this → exploit known behavior

    def __init__(
        self,
        d_model: int       = 512,
        n_action_dims: int = 32,
        reward_scale: float = 0.5,
        forward_weight: float = 0.5,
        novelty_weight: float = 0.5,
    ):
        self.forward_model = ForwardModel(d_model, n_action_dims)
        self.inverse_model = InverseModel(d_model, n_action_dims)
        self.novelty_mem   = NoveltyMemory(d_model)
        self.reward_scale  = reward_scale
        self.forward_w     = forward_weight
        self.novelty_w     = novelty_weight
        self.d_model       = d_model
        self.n_action_dims = n_action_dims

        self._history: Deque[CuriositySignal] = deque(maxlen=500)
        self._optimizer = torch.optim.Adam(
            list(self.forward_model.parameters()) +
            list(self.inverse_model.parameters()),
            lr=1e-4,
        )

    @torch.no_grad()
    def compute_curiosity(
        self,
        state:      torch.Tensor,      # (d_model,)
        action:     torch.Tensor,      # (n_action_dims,)
        next_state: torch.Tensor,      # (d_model,)
    ) -> CuriositySignal:
        """
        Compute the curiosity reward for a state transition.

        High curiosity = robot was very surprised by what happened.
        This should happen when:
          - The robot visits a new place / sees a new object
          - An action produced an unexpected result
          - Something unusual happened in the environment
        """
        self.forward_model.eval()

        state_b      = state.unsqueeze(0)
        action_b     = action.unsqueeze(0)
        next_state_b = next_state.unsqueeze(0)

        # Prediction error (how surprised was the forward model?)
        pred_error = float(
            self.forward_model.prediction_error(state_b, action_b, next_state_b)
        )
        pred_error_norm = min(1.0, pred_error / 10.0)  # Normalise

        # Novelty (have we been in this state before?)
        novelty = self.novelty_mem.novelty(next_state)

        # Combined curiosity reward
        reward = (self.forward_w  * pred_error_norm +
                  self.novelty_w  * novelty) * self.reward_scale

        # Action recommendation
        if novelty > self.EXPLORE_THRESHOLD:
            action_rec = "explore"
        elif novelty < self.EXPLOIT_THRESHOLD:
            action_rec = "exploit"
        elif pred_error_norm > 0.6:
            action_rec = "approach"
        else:
            action_rec = "exploit"

        signal = CuriositySignal(
            intrinsic_reward=round(reward, 4),
            prediction_error=round(pred_error_norm, 4),
            novelty_score=round(novelty, 4),
            recommended_action=action_rec,
        )

        # Record visit
        self.novelty_mem.record_visit(next_state)
        self._history.append(signal)

        return signal

    def learn(
        self,
        state:      torch.Tensor,
        action:     torch.Tensor,
        next_state: torch.Tensor,
    ) -> float:
        """
        Train the forward and inverse models on this experience.
        Called periodically during robot operation.
        Returns training loss.
        """
        self.forward_model.train()
        self.inverse_model.train()
        self._optimizer.zero_grad()

        s  = state.unsqueeze(0)
        a  = action.unsqueeze(0)
        ns = next_state.unsqueeze(0)

        # Forward model loss: predict next state
        pred_ns   = self.forward_model(s, a)
        fwd_loss  = F.mse_loss(pred_ns, ns.detach())

        # Inverse model loss: predict action from transition
        pred_act  = self.inverse_model(s, ns)
        inv_loss  = F.mse_loss(pred_act, a.detach())

        loss = 0.5 * fwd_loss + 0.5 * inv_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.inverse_model.parameters(), 1.0)
        self._optimizer.step()

        return float(loss)

    def recent_curiosity_level(self, window: int = 20) -> float:
        """Average curiosity reward over the last N steps."""
        recent = list(self._history)[-window:]
        if not recent:
            return 0.0
        return float(np.mean([s.intrinsic_reward for s in recent]))

    def should_explore(self) -> bool:
        """Is the robot currently in a high-curiosity state?"""
        if not self._history:
            return True
        return self._history[-1].novelty_score > self.EXPLORE_THRESHOLD

    def stats(self) -> dict:
        return {
            "avg_curiosity":  round(self.recent_curiosity_level(), 4),
            "should_explore": self.should_explore(),
            **self.novelty_mem.exploration_stats(),
        }


# ─────────────────────────────────────────────────────────────────────
#  Attention spotlight (which part of the scene to focus on)
# ─────────────────────────────────────────────────────────────────────

class AttentionSpotlight:
    """
    Determines WHERE the robot should direct its attention next.
    Combines curiosity, emotion, and task context.

    In a visual scene, this would point the camera or head
    toward the most interesting / relevant region.
    """

    def __init__(self, d_model: int = 512):
        self.d_model = d_model
        self._focus_history: List[str] = []

    def select_focus(
        self,
        patch_tokens:    Optional[torch.Tensor],    # (n_patches, d_model)
        curiosity:       CuriositySignal,
        emotion_weights: dict,
        task_embedding:  Optional[torch.Tensor] = None,
    ) -> Tuple[int, float]:
        """
        Select which patch / region to focus on.

        Priority:
          1. High-pain patches (safety first)
          2. Novel patches (curiosity)
          3. Task-relevant patches (goal-directed)
          4. Motion patches (salient)

        Returns: (patch_index, attention_weight)
        """
        if patch_tokens is None:
            return 0, 0.0

        n_patches = patch_tokens.shape[0]

        # Compute saliency per patch
        with torch.no_grad():
            # Spatial novelty: variance = how different is this patch from others?
            mean_patch = patch_tokens.mean(0, keepdim=True)
            saliency   = (patch_tokens - mean_patch).pow(2).mean(-1)  # (n_patches,)

            # If we have a task embedding, boost task-relevant patches
            if task_embedding is not None:
                relevance = F.cosine_similarity(
                    patch_tokens,
                    task_embedding.unsqueeze(0).expand(n_patches, -1),
                    dim=-1,
                )
                saliency = saliency + relevance * 0.3

            # Emotion modulation: fear → focus on most salient (potential threat)
            if emotion_weights.get("avoid", 0) > 0.5:
                pass   # Already selecting most salient
            # Curiosity → boost novel patches
            if curiosity.recommended_action == "explore":
                saliency = saliency * (1 + curiosity.novelty_score)

            best_patch  = int(saliency.argmax())
            best_weight = float(saliency.max() / (saliency.sum() + 1e-8))

        return best_patch, best_weight
