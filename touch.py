"""
ChimeraRobot Touch System
--------------------------
Processes tactile sensor data from the robot's hands and body.

Sensors per fingertip:
  • 4 pressure points (is it being squeezed?)
  • 1 temperature sensor (is it hot/cold?)
  • 1 contact flag (touching anything?)

Outputs:
  • Touch embedding (64-dim)
  • Pain signal (0–1) — triggers FEAR emotion when high
  • Texture estimate (rough/smooth/wet/sticky)
  • Grip quality (0–1) — is the grasp stable?)
  • Contact map (which fingers, which parts)

This is how a robot knows:
  "I'm holding something fragile, be gentle"
  "This is too hot, let go!"
  "I'm slipping, grip harder"
  "Something touched my back — turn around"
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_config import TouchConfig


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

TEXTURE_CLASSES = ["smooth", "rough", "soft", "hard", "wet", "sticky", "unknown"]
N_TEXTURE_CLASSES = len(TEXTURE_CLASSES)

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


@dataclass
class FingerState:
    name:       str
    contact:    bool            # Is finger touching something?
    pressure:   List[float]     # Pressure at each sensor point (0–1)
    temperature: float          # Celsius
    is_painful: bool            # Force exceeds comfort threshold?

    @property
    def mean_pressure(self) -> float:
        return float(np.mean(self.pressure)) if self.pressure else 0.0

    @property
    def max_pressure(self) -> float:
        return float(np.max(self.pressure)) if self.pressure else 0.0


@dataclass
class HandState:
    hand:       str             # "left" | "right"
    fingers:    List[FingerState]
    palm_contact: bool
    palm_pressure: float        # 0–1
    wrist_torque: float         # Newton-metres

    @property
    def n_fingers_touching(self) -> int:
        return sum(1 for f in self.fingers if f.contact)

    @property
    def grip_force(self) -> float:
        return float(np.mean([f.mean_pressure for f in self.fingers]))


@dataclass
class TouchScene:
    left_hand:   HandState
    right_hand:  HandState
    embedding:   torch.Tensor       # (feature_dim,) fused touch embedding
    pain_level:  float              # 0 = no pain, 1 = max pain
    grip_quality: float             # 0 = dropping, 1 = secure
    texture:     str                # Estimated texture
    texture_probs: Dict[str, float]
    contact_map: np.ndarray         # (n_fingers * 2,) binary contact flags
    timestamp:   float = 0.0


# ─────────────────────────────────────────────────────────────────────
#  Touch encoder neural network
# ─────────────────────────────────────────────────────────────────────

class TouchEncoder(nn.Module):
    """
    Encodes raw tactile sensor readings into a compact embedding.

    Input features per hand:
      - n_fingers × n_sensors_per_finger pressure values
      - n_fingers temperature values
      - n_fingers contact flags
      - palm contact + pressure
      = 5 × (4 + 1 + 1) + 2 = 32 features per hand
      Total input: 64 features (both hands)

    Output: 64-dim touch embedding + texture classification + pain signal
    """

    def __init__(self, config: TouchConfig, output_dim: int = 64):
        super().__init__()
        n_per_hand = (config.n_fingers * (config.n_sensors_per_finger + 2) + 2)
        input_dim  = n_per_hand * config.n_hands

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Texture classifier
        self.texture_head = nn.Linear(output_dim, N_TEXTURE_CLASSES)

        # Pain estimator (scalar)
        self.pain_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Grip quality estimator
        self.grip_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _hand_to_vector(self, hand: HandState, config: TouchConfig) -> np.ndarray:
        """Flatten HandState into a numpy feature vector."""
        feats = []
        for finger in hand.fingers:
            feats.extend(finger.pressure[:config.n_sensors_per_finger])
            # Pad if fewer sensor readings than expected
            while len(feats) % config.n_sensors_per_finger != 0:
                feats.append(0.0)
            feats.append(finger.temperature / 100.0)   # Normalize to [0,1]
            feats.append(float(finger.contact))
        feats.append(float(hand.palm_contact))
        feats.append(hand.palm_pressure)
        return np.array(feats, dtype=np.float32)

    def forward(
        self, left_vec: torch.Tensor, right_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            left_vec:   (B, n_per_hand)
            right_vec:  (B, n_per_hand)
        Returns:
            embedding, texture_logits, pain, grip_quality
        """
        x   = torch.cat([left_vec, right_vec], dim=-1)
        emb = self.encoder(x)
        texture = self.texture_head(emb)
        pain    = self.pain_head(emb)
        grip    = self.grip_head(emb)
        return emb, texture, pain, grip

    @torch.no_grad()
    def encode_states(
        self, left: HandState, right: HandState, config: TouchConfig
    ) -> Tuple[torch.Tensor, str, float, float]:
        """Full encode from HandState objects."""
        self.eval()
        lv = torch.from_numpy(self._hand_to_vector(left,  config)).unsqueeze(0)
        rv = torch.from_numpy(self._hand_to_vector(right, config)).unsqueeze(0)

        emb, tex_logits, pain_t, grip_t = self.forward(lv, rv)

        tex_probs = F.softmax(tex_logits, dim=-1).squeeze(0).tolist()
        tex_class = TEXTURE_CLASSES[int(tex_logits.argmax(-1))]

        return (
            emb.squeeze(0),
            tex_class,
            float(pain_t.squeeze()),
            float(grip_t.squeeze()),
        )


# ─────────────────────────────────────────────────────────────────────
#  Pain processor — maps sensor readings to emotional signals
# ─────────────────────────────────────────────────────────────────────

class PainProcessor:
    """
    Converts raw sensor readings into a pain signal [0, 1].
    Pain is the robot's protective mechanism — it triggers FEAR and
    teaches the robot not to repeat damaging actions.

    Sources of pain:
      - Force limit exceeded (squeezing too hard / being squeezed)
      - Temperature too high or too low
      - Collision (sudden deceleration in IMU)
      - Joint torque limit exceeded
    """

    def __init__(self, config: TouchConfig):
        self.config       = config
        self.max_force    = config.max_force_n
        self.temp_min     = config.temperature_range[0]
        self.temp_max     = config.temperature_range[1]
        self.pain_history: List[Tuple[float, float]] = []   # (time, pain)

    def compute_pain(self, left: HandState, right: HandState) -> float:
        """Aggregate pain from all sensors. Returns 0.0–1.0."""
        pains: List[float] = []

        for hand in [left, right]:
            for finger in hand.fingers:
                # Pressure pain: linear above 80% of max
                for p in finger.pressure:
                    if p > 0.8:
                        pains.append((p - 0.8) / 0.2)
                # Temperature pain: pain at extremes
                t = finger.temperature
                if t > 50:
                    pains.append(min(1.0, (t - 50) / 40))
                elif t < 5:
                    pains.append(min(1.0, (5 - t) / 10))

            # Wrist torque pain
            if hand.wrist_torque > 0.9:
                pains.append((hand.wrist_torque - 0.9) / 0.1)

        pain = max(pains) if pains else 0.0
        self.pain_history.append((time.time(), pain))
        if len(self.pain_history) > 100:
            self.pain_history.pop(0)
        return round(pain, 3)

    def cumulative_pain(self, window_s: float = 5.0) -> float:
        """Average pain over the last `window_s` seconds."""
        now = time.time()
        recent = [p for t, p in self.pain_history if now - t < window_s]
        return float(np.mean(recent)) if recent else 0.0


# ─────────────────────────────────────────────────────────────────────
#  Full touch system
# ─────────────────────────────────────────────────────────────────────

class TouchSystem:
    """
    Complete touch processing pipeline.
    Reads sensor data → HandState → TouchScene with embeddings.
    """

    def __init__(self, config: TouchConfig):
        self.config    = config
        self.encoder   = TouchEncoder(config)
        self.pain_proc = PainProcessor(config)

    def process(self, left: HandState, right: HandState) -> TouchScene:
        """
        Args:
            left, right: HandState from hardware / simulator

        Returns: TouchScene with embedding, pain, grip quality, texture
        """
        emb, texture, pain_nn, grip = self.encoder.encode_states(left, right, self.config)

        # Compute pain from raw sensor values (more reliable than NN at start)
        pain_raw = self.pain_proc.compute_pain(left, right)
        pain = max(float(pain_nn), pain_raw)

        # Contact map
        left_contacts  = [float(f.contact) for f in left.fingers]
        right_contacts = [float(f.contact) for f in right.fingers]
        contact_map    = np.array(left_contacts + right_contacts, dtype=np.float32)

        return TouchScene(
            left_hand=left,
            right_hand=right,
            embedding=emb,
            pain_level=pain,
            grip_quality=float(grip),
            texture=texture,
            texture_probs={c: 0.0 for c in TEXTURE_CLASSES},  # Fill if needed
            contact_map=contact_map,
            timestamp=time.time(),
        )


# ─────────────────────────────────────────────────────────────────────
#  Mock hardware
# ─────────────────────────────────────────────────────────────────────

class MockTouchSensor:
    """
    Generates synthetic touch data for simulation.
    Simulates picking up an object around t=3s.
    """

    def __init__(self, config: TouchConfig):
        self.config = config
        self._t     = 0

    def read(self) -> Tuple[HandState, HandState]:
        self._t += 1
        holding = 30 < self._t < 80  # Simulate holding object

        def make_finger(name: str, active: bool) -> FingerState:
            pressure = [0.6 if active else 0.02] * self.config.n_sensors_per_finger
            if active:
                pressure[0] += np.random.uniform(-0.05, 0.05)
            temp = 22.0 + np.random.normal(0, 0.5)
            return FingerState(
                name=name,
                contact=active,
                pressure=[max(0, min(1, p)) for p in pressure],
                temperature=temp,
                is_painful=False,
            )

        fingers = [make_finger(n, holding) for n in FINGER_NAMES]

        def make_hand(side: str) -> HandState:
            return HandState(
                hand=side,
                fingers=fingers.copy(),
                palm_contact=holding,
                palm_pressure=0.3 if holding else 0.0,
                wrist_torque=0.2 if holding else 0.0,
            )

        return make_hand("left"), make_hand("right")
