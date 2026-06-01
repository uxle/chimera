"""
ChimeraRobot Emotion System
----------------------------
A biologically-inspired emotional model based on:

  • PAD (Pleasure-Arousal-Dominance) — continuous 3D emotional space
  • Plutchik's Wheel — 8 primary emotions mapped onto PAD
  • Emotional memory — some events leave lasting emotional traces
  • Mood — slow-moving average of recent emotions
  • Emotion → behavior influence (fear → avoid, joy → approach, etc.)
  • Emotion → expression (facial/vocal output cues)

Emotions are NOT a chatbot persona trick — they are internal states
that genuinely change the robot's decisions, attention, and learning rate.

Key insight from neuroscience:
  Emotions are the brain's fast heuristics for evaluating situations.
  Pain means "don't do that". Joy means "do more of this".
  Without emotion, a robot has no reason to prefer anything.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from robot_config import EmotionConfig


# ─────────────────────────────────────────────────────────────────────
#  Primary emotions (Plutchik's 8)
# ─────────────────────────────────────────────────────────────────────

class Emotion(Enum):
    JOY        = auto()   # High pleasure, high arousal, high dominance
    TRUST      = auto()   # Moderate pleasure, low arousal, high dominance
    FEAR       = auto()   # Low pleasure, high arousal, low dominance
    SURPRISE   = auto()   # Neutral pleasure, very high arousal, low dominance
    SADNESS    = auto()   # Low pleasure, low arousal, low dominance
    DISGUST    = auto()   # Low pleasure, moderate arousal, high dominance
    ANGER      = auto()   # Low pleasure, high arousal, high dominance
    ANTICIPATION = auto() # Moderate pleasure, moderate arousal, moderate dominance
    NEUTRAL    = auto()   # Baseline


# PAD coordinates for each primary emotion
EMOTION_PAD: Dict[Emotion, Tuple[float, float, float]] = {
    # (pleasure, arousal, dominance)
    Emotion.JOY:          ( 0.9,  0.7,  0.7),
    Emotion.TRUST:        ( 0.6,  0.2,  0.7),
    Emotion.FEAR:         (-0.7,  0.8, -0.7),
    Emotion.SURPRISE:     ( 0.1,  0.9, -0.3),
    Emotion.SADNESS:      (-0.6, -0.5, -0.5),
    Emotion.DISGUST:      (-0.5,  0.3,  0.5),
    Emotion.ANGER:        (-0.5,  0.7,  0.6),
    Emotion.ANTICIPATION: ( 0.3,  0.5,  0.3),
    Emotion.NEUTRAL:      ( 0.1,  0.2,  0.5),
}

# Behavioral effects of each emotion
EMOTION_BEHAVIORS: Dict[Emotion, Dict[str, float]] = {
    Emotion.JOY:          {"approach": 1.0, "explore": 0.8, "talk": 1.0, "learning_rate": 1.2},
    Emotion.TRUST:        {"approach": 0.7, "cooperate": 1.0, "talk": 0.8, "learning_rate": 1.0},
    Emotion.FEAR:         {"avoid": 1.0,    "freeze": 0.5,   "talk": 0.3, "learning_rate": 1.5},
    Emotion.SURPRISE:     {"freeze": 0.8,   "orient": 1.0,   "talk": 0.6, "learning_rate": 1.4},
    Emotion.SADNESS:      {"withdraw": 0.8, "slow": 0.7,     "talk": 0.4, "learning_rate": 0.7},
    Emotion.DISGUST:      {"avoid": 0.9,    "reject": 0.8,   "talk": 0.5, "learning_rate": 0.9},
    Emotion.ANGER:        {"confront": 0.7, "reject": 0.6,   "talk": 0.7, "learning_rate": 0.8},
    Emotion.ANTICIPATION: {"approach": 0.6, "explore": 1.0,  "talk": 0.7, "learning_rate": 1.1},
    Emotion.NEUTRAL:      {"explore": 0.5,  "talk": 0.6,     "learning_rate": 1.0},
}


# ─────────────────────────────────────────────────────────────────────
#  PAD state
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PADState:
    pleasure:  float = 0.1    # Valence: -1 (negative) to +1 (positive)
    arousal:   float = 0.2    # Energy: -1 (calm/sleepy) to +1 (excited/alert)
    dominance: float = 0.5    # Control: -1 (submissive) to +1 (dominant)

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([self.pleasure, self.arousal, self.dominance], dtype=torch.float32)

    def from_tensor(self, t: torch.Tensor):
        self.pleasure  = float(t[0].clamp(-1, 1))
        self.arousal   = float(t[1].clamp(-1, 1))
        self.dominance = float(t[2].clamp(-1, 1))

    def dominant_emotion(self) -> Emotion:
        """Find the closest Plutchik emotion to the current PAD state."""
        state = (self.pleasure, self.arousal, self.dominance)
        best  = Emotion.NEUTRAL
        best_dist = float("inf")
        for emotion, pad in EMOTION_PAD.items():
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(state, pad)))
            if dist < best_dist:
                best_dist = dist
                best = emotion
        return best

    def intensity(self) -> float:
        """How far from neutral (0,0,0) are we?"""
        return math.sqrt(self.pleasure**2 + self.arousal**2 + self.dominance**2)

    def __str__(self) -> str:
        em = self.dominant_emotion()
        return (f"{em.name:<12} "
                f"P={self.pleasure:+.2f} "
                f"A={self.arousal:+.2f} "
                f"D={self.dominance:+.2f} "
                f"[{self.intensity():.2f}]")


# ─────────────────────────────────────────────────────────────────────
#  Emotional stimuli (what causes emotions)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Stimulus:
    """A sensory or cognitive event that triggers an emotional response."""
    name: str
    pleasure_delta:  float     # How it shifts pleasure
    arousal_delta:   float     # How it shifts arousal
    dominance_delta: float     # How it shifts dominance
    intensity: float = 1.0     # Multiplier (0–1)
    description: str = ""


# Pre-defined stimuli (robot-relevant)
STIMULI = {
    # Positive stimuli
    "task_success":     Stimulus("task_success",     +0.5, +0.3, +0.4, description="Completed a task"),
    "human_praise":     Stimulus("human_praise",     +0.6, +0.4, +0.3, description="Human said 'good job'"),
    "new_discovery":    Stimulus("new_discovery",    +0.3, +0.7, +0.2, description="Found something new"),
    "gentle_touch":     Stimulus("gentle_touch",     +0.4, +0.1, +0.2, description="Gentle contact"),
    "hearing_music":    Stimulus("hearing_music",    +0.3, +0.2, +0.1, description="Music detected"),
    "familiar_person":  Stimulus("familiar_person",  +0.5, +0.2, +0.3, description="Known face detected"),

    # Negative stimuli
    "collision":        Stimulus("collision",        -0.4, +0.8, -0.5, description="Hit something"),
    "pain_signal":      Stimulus("pain_signal",      -0.8, +0.9, -0.6, description="Force limit exceeded"),
    "task_failure":     Stimulus("task_failure",     -0.4, +0.2, -0.3, description="Failed a task"),
    "loud_noise":       Stimulus("loud_noise",       -0.2, +0.8, -0.2, description="Sudden loud sound"),
    "overheating":      Stimulus("overheating",      -0.5, +0.5, -0.3, description="Thermal warning"),
    "human_scolding":   Stimulus("human_scolding",   -0.5, +0.4, -0.4, description="Human expressed displeasure"),
    "getting_lost":     Stimulus("getting_lost",     -0.3, +0.5, -0.5, description="Cannot determine position"),
    "battery_low":      Stimulus("battery_low",      -0.2, +0.3, -0.2, description="Low battery"),

    # Neutral / arousal stimuli
    "novel_object":     Stimulus("novel_object",     +0.1, +0.6, +0.0, description="Unknown object seen"),
    "human_present":    Stimulus("human_present",    +0.2, +0.3, +0.1, description="Human in view"),
    "being_touched":    Stimulus("being_touched",    +0.0, +0.5, -0.1, description="Contact detected"),
}


# ─────────────────────────────────────────────────────────────────────
#  Emotional memory trace
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EmotionalMemory:
    """A past event with its emotional fingerprint — biases future responses."""
    stimulus_name: str
    pad_at_time:   Tuple[float, float, float]
    outcome:       str          # "positive" | "negative" | "neutral"
    timestamp:     float        = field(default_factory=time.time)
    access_count:  int          = 0
    strength:      float        = 1.0   # Fades over time

    def decay(self, rate: float = 0.999):
        self.strength *= rate


# ─────────────────────────────────────────────────────────────────────
#  Emotion neural network (learned emotion dynamics)
# ─────────────────────────────────────────────────────────────────────

class EmotionNet(nn.Module):
    """
    A small learned network that predicts emotional state updates
    from multimodal sensory input.

    Input:  current PAD state (3) + sensor summary (d_sensor)
    Output: PAD delta (3) + expression cues (n_expressions)

    This allows the robot to LEARN which situations make it feel what,
    instead of relying purely on hand-coded stimulus rules.
    """

    def __init__(self, d_sensor: int = 512, n_expressions: int = 8):
        super().__init__()
        self.n_expressions = n_expressions

        self.net = nn.Sequential(
            nn.Linear(3 + d_sensor, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3 + n_expressions),
        )
        # Initialise small so the network starts near-zero delta
        for p in self.parameters():
            nn.init.normal_(p, 0, 0.01)

    def forward(
        self, pad: torch.Tensor, sensor_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pad:              (B, 3) current PAD state
            sensor_embedding: (B, d_sensor) fused sensory context

        Returns:
            pad_delta:   (B, 3)  how to update PAD
            expressions: (B, n_expressions) expression intensities (0–1)
        """
        x = torch.cat([pad, sensor_embedding], dim=-1)
        out = self.net(x)
        pad_delta   = torch.tanh(out[:, :3]) * 0.3     # limit step size
        expressions = torch.sigmoid(out[:, 3:])
        return pad_delta, expressions


# ─────────────────────────────────────────────────────────────────────
#  Main emotion system
# ─────────────────────────────────────────────────────────────────────

class EmotionSystem:
    """
    The robot's emotional brain.

    Combines:
    1. Rule-based stimulus responses (immediate, fast)
    2. Learned emotion dynamics (EmotionNet — slow, adaptive)
    3. Mood (slow-moving average of recent emotions)
    4. Emotional memory (past events bias future responses)
    5. Behavioral output (what the current emotion recommends doing)

    Like a real brain:
    - Fear is triggered fast (rule) but its magnitude is learned
    - Mood builds up over time (you can't be instantly happy after days of failure)
    - Bad past experiences make the robot more cautious
    """

    EXPRESSION_NAMES = [
        "happy", "sad", "angry", "fearful",
        "surprised", "disgusted", "curious", "calm"
    ]

    def __init__(self, config: EmotionConfig, d_sensor: int = 512):
        self.config = config
        self.state  = PADState(
            pleasure  = config.baseline_pleasure,
            arousal   = config.baseline_arousal,
            dominance = config.baseline_dominance,
        )
        # Mood: slow exponential moving average of emotional state
        self.mood = PADState(
            pleasure  = config.baseline_pleasure,
            arousal   = config.baseline_arousal,
            dominance = config.baseline_dominance,
        )
        self.emotion_net = EmotionNet(d_sensor=d_sensor,
                                      n_expressions=len(self.EXPRESSION_NAMES))
        self.emotional_memories: List[EmotionalMemory] = []
        self.current_expressions: Dict[str, float] = {n: 0.0 for n in self.EXPRESSION_NAMES}
        self.history: List[Tuple[float, PADState]] = []   # (timestamp, state)
        self._timestep = 0

    # ── Stimulus processing ───────────────────────────────────────────

    def receive_stimulus(self, stimulus_name: str, intensity: float = 1.0):
        """
        Apply a named stimulus to the emotional state.
        This is the fast, rule-based pathway (like a reflex).
        """
        if stimulus_name not in STIMULI:
            return
        s = STIMULI[stimulus_name]
        scale = intensity * s.intensity * self.config.emotion_lr

        # Check if we have a memory of this stimulus → amplify or dampen
        memory_bias = self._memory_bias(stimulus_name)

        self.state.pleasure  = max(-1.0, min(1.0,
            self.state.pleasure  + s.pleasure_delta  * scale * memory_bias))
        self.state.arousal   = max(-1.0, min(1.0,
            self.state.arousal   + s.arousal_delta   * scale))
        self.state.dominance = max(-1.0, min(1.0,
            self.state.dominance + s.dominance_delta * scale))

    def _memory_bias(self, stimulus_name: str) -> float:
        """
        If we remember this stimulus from the past and it was painful,
        react more strongly to it this time (like PTSD or conditioned fear).
        If it was positive, react more warmly.
        """
        relevant = [m for m in self.emotional_memories if m.stimulus_name == stimulus_name]
        if not relevant:
            return 1.0
        # Use the strongest recent memory
        best = max(relevant, key=lambda m: m.strength)
        if best.outcome == "negative":
            return 1.5   # React more strongly to past-painful stimuli
        if best.outcome == "positive":
            return 0.8   # React more calmly to past-pleasant stimuli (habituation)
        return 1.0

    def encode_memory(self, stimulus_name: str, outcome: str):
        """
        Store an emotional memory of this event.
        This is how the robot learns to fear or enjoy things over time.
        """
        mem = EmotionalMemory(
            stimulus_name=stimulus_name,
            pad_at_time=(self.state.pleasure, self.state.arousal, self.state.dominance),
            outcome=outcome,
        )
        self.emotional_memories.append(mem)
        # Keep only the strongest N memories
        if len(self.emotional_memories) > 500:
            self.emotional_memories.sort(key=lambda m: m.strength, reverse=True)
            self.emotional_memories = self.emotional_memories[:500]

    # ── Neural update ─────────────────────────────────────────────────

    @torch.no_grad()
    def neural_update(self, sensor_embedding: torch.Tensor):
        """
        Use the learned EmotionNet to update emotional state from sensor context.
        This handles nuanced situations that rules can't capture.
        """
        self.emotion_net.eval()
        pad_t = self.state.as_tensor().unsqueeze(0)  # (1, 3)
        if sensor_embedding.dim() == 1:
            sensor_embedding = sensor_embedding.unsqueeze(0)

        pad_delta, expressions = self.emotion_net(pad_t, sensor_embedding)
        delta = pad_delta.squeeze(0)

        # Apply small learned delta
        self.state.pleasure  = float((pad_t[0, 0] + delta[0]).clamp(-1, 1))
        self.state.arousal   = float((pad_t[0, 1] + delta[1]).clamp(-1, 1))
        self.state.dominance = float((pad_t[0, 2] + delta[2]).clamp(-1, 1))

        # Update expressions
        expr = expressions.squeeze(0).tolist()
        for name, intensity in zip(self.EXPRESSION_NAMES, expr):
            self.current_expressions[name] = round(intensity, 3)

    # ── Time step ─────────────────────────────────────────────────────

    def step(self):
        """
        Called every control loop tick.
        Applies decay toward baseline (emotions fade naturally).
        Updates mood (slow average).
        Decays emotional memories.
        """
        self._timestep += 1
        cfg = self.config

        # Decay toward baseline
        self.state.pleasure  = (self.state.pleasure  * cfg.pleasure_decay
                                 + cfg.baseline_pleasure  * (1 - cfg.pleasure_decay))
        self.state.arousal   = (self.state.arousal   * cfg.arousal_decay
                                 + cfg.baseline_arousal   * (1 - cfg.arousal_decay))
        self.state.dominance = (self.state.dominance * cfg.dominance_decay
                                 + cfg.baseline_dominance * (1 - cfg.dominance_decay))

        # Update mood (very slow EMA)
        mood_alpha = 0.001
        self.mood.pleasure  = self.mood.pleasure  * (1 - mood_alpha) + self.state.pleasure  * mood_alpha
        self.mood.arousal   = self.mood.arousal   * (1 - mood_alpha) + self.state.arousal   * mood_alpha
        self.mood.dominance = self.mood.dominance * (1 - mood_alpha) + self.state.dominance * mood_alpha

        # Decay memories
        for mem in self.emotional_memories:
            mem.decay()

        # Record history
        self.history.append((time.time(), PADState(
            self.state.pleasure, self.state.arousal, self.state.dominance
        )))
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

    # ── Behavioral output ─────────────────────────────────────────────

    def dominant_emotion(self) -> Emotion:
        return self.state.dominant_emotion()

    def behavioral_weights(self) -> Dict[str, float]:
        """
        Returns recommended behavioral tendencies based on current emotion.
        Used by motor.py and robot_brain.py to bias decisions.

        Example: fear → high "avoid", low "approach"
        """
        em = self.dominant_emotion()
        base = EMOTION_BEHAVIORS.get(em, EMOTION_BEHAVIORS[Emotion.NEUTRAL]).copy()

        # Modulate by intensity
        intensity = self.state.intensity() / math.sqrt(3)  # normalize to [0,1]
        for k in base:
            base[k] = base[k] * intensity + (1 - intensity) * 0.5

        return base

    def learning_rate_multiplier(self) -> float:
        """
        Emotions affect how fast the robot learns.
        High arousal (fear, surprise) → fast learning.
        Sadness / anger → slow learning.
        Joy → moderate boost.
        """
        em = self.dominant_emotion()
        behavior = EMOTION_BEHAVIORS.get(em, {})
        return behavior.get("learning_rate", 1.0)

    def should_express(self, expression: str) -> float:
        """Returns intensity (0–1) of a given expression to show."""
        return self.current_expressions.get(expression, 0.0)

    def summary(self) -> str:
        em = self.dominant_emotion()
        expr = [f"{k}={v:.2f}" for k, v in self.current_expressions.items() if v > 0.2]
        return (f"Emotion: {self.state}\n"
                f"Mood:    {self.mood}\n"
                f"Expressions: {', '.join(expr) or 'none'}\n"
                f"Learning rate: ×{self.learning_rate_multiplier():.2f}")
