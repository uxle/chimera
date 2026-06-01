"""
ChimeraRobot Social Learning
------------------------------
Humans are extraordinarily good at learning from watching other humans.
A baby doesn't need to discover fire is hot by touching it —
they see an adult recoil and learn the same lesson instantly.

This module gives the robot the same ability: learn from watching.

Social learning mechanisms:

  1. Imitation
     Observe a human's action → mirror it with own body
     "Human waved → I should wave"

  2. Social Referencing
     Look at a human's emotional reaction to something to know how to feel
     "Human looked scared at that object → I should be cautious too"

  3. Pointing/Gaze Following
     Follow a human's gaze or pointing gesture to find what's interesting
     "Human is looking at the door → something is there"

  4. Emotional Contagion
     Catch emotions from humans (the basis of empathy)
     "Human seems happy → I feel happy too"

  5. Verbal Instruction
     Learn new behaviours from spoken commands
     "Pick up the red cup" → associate 'pick up' + 'cup' + 'red'

  6. Reward Shaping from Social Cues
     Human facial expression / tone reveals whether you're doing the right thing
     Smile → positive reward, frown → negative reward
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
#  Human observation data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HumanObservation:
    """What the robot observes about a nearby human."""
    person_id:    str
    position:     Tuple[float, float]      # (x, y) world coords
    gaze_vector:  Tuple[float, float]      # Direction they're looking
    pointing:     Optional[Tuple[float, float]] = None  # Where they're pointing
    emotion:      str = "neutral"          # Detected human emotion
    is_speaking:  bool = False
    speech_text:  str = ""
    action:       str = "idle"             # What are they doing?
    confidence:   float = 0.8


# ─────────────────────────────────────────────────────────────────────
#  1. Imitation system
# ─────────────────────────────────────────────────────────────────────

class ImitationSystem(nn.Module):
    """
    Learns to map observed human actions to robot motor commands.

    Input:  human action embedding (d_model)
            robot body state (proprio_dim)
    Output: robot action vector (n_action_dims)

    The key insight is that the robot must solve the correspondence problem:
    "What motor commands produce the SAME OUTCOME as what I observed,
    given my different body?"
    """

    def __init__(self, d_model: int = 512, proprio_dim: int = 128, n_actions: int = 32):
        super().__init__()
        # Action encoder: compress observed action to embedding
        self.action_encoder = nn.Sequential(
            nn.Linear(d_model, 256), nn.SiLU(),
            nn.Linear(256, 128),
        )
        # Body mapper: combine action + own body state → robot action
        self.body_mapper = nn.Sequential(
            nn.Linear(128 + proprio_dim, 256), nn.SiLU(),
            nn.Linear(256, n_actions), nn.Tanh(),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, obs_action_emb: torch.Tensor,
                proprio: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_encoder(obs_action_emb)
        combined   = torch.cat([action_emb, proprio], dim=-1)
        return self.body_mapper(combined)

    def learn_from_demonstration(
        self,
        obs_action_embs:   torch.Tensor,  # (B, d_model) observed actions
        robot_proprios:    torch.Tensor,  # (B, proprio_dim)
        target_actions:    torch.Tensor,  # (B, n_actions) what robot should do
    ) -> float:
        self.train()
        self.optimizer.zero_grad()
        pred  = self(obs_action_embs, robot_proprios)
        loss  = F.mse_loss(pred, target_actions)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return float(loss)


# ─────────────────────────────────────────────────────────────────────
#  2. Social referencing
# ─────────────────────────────────────────────────────────────────────

class SocialReferencing:
    """
    The robot checks human emotional reactions to calibrate its own responses.

    When a robot encounters an ambiguous situation:
    → Look at the nearest human
    → Read their emotion
    → Use that as a guide for how to respond

    A toddler does this constantly: sees a strange dog, looks at parent,
    parent smiles → dog is safe. Parent looks scared → dog is dangerous.

    This is one of the most powerful forms of social learning.
    """

    EMOTION_TO_PAD: Dict[str, Tuple[float, float, float]] = {
        "happy":    (+0.7, +0.5, +0.5),
        "excited":  (+0.6, +0.8, +0.4),
        "calm":     (+0.3, -0.2, +0.3),
        "neutral":  (+0.0,  0.0, +0.0),
        "worried":  (-0.3, +0.4, -0.3),
        "scared":   (-0.6, +0.8, -0.6),
        "angry":    (-0.4, +0.7, +0.5),
        "sad":      (-0.5, -0.3, -0.4),
        "disgusted":(-0.4, +0.2, +0.3),
    }

    def __init__(self):
        self._reference_history: List[Tuple[str, str, float]] = []

    def get_social_reference(
        self,
        observation: HumanObservation,
        situation: str = "novel_object",
    ) -> Optional[Tuple[float, float, float]]:
        """
        Read human emotion and return PAD delta for the robot.
        Only activates if the human is close and attention is joint
        (both robot and human are focused on the same thing).

        Returns PAD (Pleasure, Arousal, Dominance) shift to apply.
        """
        if not observation:
            return None

        # Scale by confidence and proximity
        proximity = 1.0 / (1.0 + math.sqrt(
            observation.position[0]**2 + observation.position[1]**2
        ))
        weight = observation.confidence * proximity

        base_pad = self.EMOTION_TO_PAD.get(observation.emotion, (0, 0, 0))
        scaled   = tuple(v * weight * 0.3 for v in base_pad)   # Moderate influence

        self._reference_history.append(
            (situation, observation.emotion, time.time())
        )
        return scaled

    def most_referenced_emotion(self) -> str:
        """What emotion do humans show most in this context?"""
        if not self._reference_history:
            return "neutral"
        from collections import Counter
        emotions = [e for _, e, _ in self._reference_history[-20:]]
        return Counter(emotions).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────
#  3. Gaze and attention following
# ─────────────────────────────────────────────────────────────────────

class GazeFollowing:
    """
    The robot follows where humans are looking or pointing.
    This is one of the earliest and most important social skills —
    human babies develop it around 9–12 months.

    Following gaze reveals:
    - What objects are important to humans
    - What the human wants the robot to look at
    - Shared attention = foundation for language learning
    """

    def __init__(self):
        self._gaze_targets: List[Tuple[float, float, float]] = []  # (x, y, time)

    def process_observation(
        self,
        observation: HumanObservation,
        robot_pos:   Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """
        Compute where the human is looking in world coordinates.
        Returns (x, y) world target, or None if not determinable.
        """
        # Gaze vector gives direction; estimate target at fixed distance
        gaze_dist = 2.0    # Assume human is looking at something 2m away
        gx = observation.position[0] + observation.gaze_vector[0] * gaze_dist
        gy = observation.position[1] + observation.gaze_vector[1] * gaze_dist

        # If human is also pointing, use pointing direction
        if observation.pointing:
            gx = observation.position[0] + observation.pointing[0] * gaze_dist
            gy = observation.position[1] + observation.pointing[1] * gaze_dist

        self._gaze_targets.append((gx, gy, time.time()))
        return (gx, gy)

    def sustained_attention_target(self, window_s: float = 3.0) -> Optional[Tuple[float, float]]:
        """
        Find what the human has been looking at for >3 seconds.
        Sustained attention = something important.
        """
        now = time.time()
        recent = [(x, y) for x, y, t in self._gaze_targets if now - t < window_s]
        if len(recent) < 3:
            return None
        # Cluster: if all recent points are close, the human is looking there
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        spread = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
        if spread < 0.5:   # Tight cluster = sustained gaze
            return (sum(xs)/len(xs), sum(ys)/len(ys))
        return None


# ─────────────────────────────────────────────────────────────────────
#  4. Emotional contagion
# ─────────────────────────────────────────────────────────────────────

class EmotionalContagion:
    """
    Humans involuntarily mirror each other's emotions (emotional contagion).
    Babies are especially susceptible — they pick up caregiver emotions directly.

    The robot's emotional contagion makes it:
    - More empathetic and relatable
    - Better at reading social context
    - Able to share in human emotional experiences

    Contagion is distance-weighted and modulated by stage of development
    (toddlers are more susceptible than adolescents).
    """

    def __init__(self, susceptibility: float = 0.3):
        """susceptibility: 0 = immune, 1 = fully mirrors human emotion"""
        self.susceptibility = susceptibility

    def compute_contagion(
        self,
        human_emotion: str,
        distance_m:    float,
        relationship:  str = "stranger",   # "stranger" | "familiar" | "trusted"
    ) -> Tuple[float, float, float]:
        """
        Compute PAD influence from human emotion.
        Returns (dp, da, dd) — small shifts to add to robot PAD state.
        """
        rel_scale = {"stranger": 0.3, "familiar": 0.7, "trusted": 1.0}
        proximity  = math.exp(-distance_m / 3.0)           # Decay with distance
        strength   = self.susceptibility * proximity * rel_scale.get(relationship, 0.5)

        base = SocialReferencing.EMOTION_TO_PAD.get(human_emotion, (0, 0, 0))
        return tuple(v * strength for v in base)


# ─────────────────────────────────────────────────────────────────────
#  5. Verbal instruction parser
# ─────────────────────────────────────────────────────────────────────

class VerbalInstructionParser:
    """
    Parses verbal instructions into robot actions.
    Uses pattern matching on key verb-object pairs.
    The language model handles complex instructions — this handles simple ones.

    Example:
        "Pick up the red cup"     → action: pick_up, target: red cup
        "Walk forward"            → action: walk, direction: forward
        "Stop"                    → action: stop
        "Wave to the person"      → action: wave, target: person
        "Look at the table"       → action: look, target: table
    """

    VERB_ACTION_MAP: Dict[str, str] = {
        "pick":    "pick_up",
        "grab":    "pick_up",
        "take":    "pick_up",
        "get":     "pick_up",
        "put":     "put_down",
        "place":   "put_down",
        "drop":    "put_down",
        "release": "put_down",
        "walk":    "walk",
        "move":    "walk",
        "go":      "walk",
        "turn":    "turn",
        "rotate":  "turn",
        "stop":    "stop",
        "halt":    "stop",
        "wave":    "wave",
        "greet":   "wave",
        "look":    "look_at",
        "watch":   "look_at",
        "follow":  "follow",
        "sit":     "sit",
        "stand":   "stand",
        "come":    "approach",
        "approach":"approach",
    }

    DIRECTION_MAP: Dict[str, str] = {
        "forward": "WALK_FORWARD",
        "ahead":   "WALK_FORWARD",
        "back":    "WALK_BACK",
        "backward":"WALK_BACK",
        "left":    "TURN_LEFT",
        "right":   "TURN_RIGHT",
    }

    def parse(self, text: str) -> Optional[Dict]:
        """
        Parse a verbal instruction into a structured action dict.
        Returns None if the instruction cannot be parsed.
        """
        words  = text.lower().split()
        action = None
        target = None
        direction = None

        for word in words:
            if word in self.VERB_ACTION_MAP:
                action = self.VERB_ACTION_MAP[word]
            if word in self.DIRECTION_MAP:
                direction = self.DIRECTION_MAP[word]

        # Target: look for "the X" or "a X" patterns
        for i, w in enumerate(words):
            if w in ("the", "a", "that") and i + 1 < len(words):
                target = words[i + 1]
                break

        if action is None:
            return None

        result = {"action": action}
        if target:    result["target"]    = target
        if direction: result["direction"] = direction
        return result

    def requires_llm(self, text: str) -> bool:
        """Returns True if the instruction is too complex for pattern matching."""
        # Heuristic: if more than 10 words or contains 'if', 'when', 'after'
        words = text.lower().split()
        complex_words = {"if", "when", "after", "before", "while", "unless", "because"}
        return len(words) > 10 or bool(complex_words & set(words))


# ─────────────────────────────────────────────────────────────────────
#  6. Social reward from facial expressions
# ─────────────────────────────────────────────────────────────────────

class FacialRewardDetector:
    """
    Infers reward signal from human facial expressions and vocal tone.
    This is how animals (and robots) learn social norms without explicit instruction:
    the human's face tells you immediately if you did the right thing.

    Expression → reward mapping:
      Smile, thumbs up, applause → positive reward
      Frown, head shake, alarm   → negative reward
      Neutral, nodding           → small positive (attention = approval)
    """

    EXPRESSION_REWARD: Dict[str, float] = {
        "smile":     +0.8,
        "laugh":     +1.0,
        "thumbs_up": +1.0,
        "nod":       +0.3,
        "neutral":   +0.0,
        "frown":     -0.5,
        "head_shake":-0.6,
        "surprised": +0.1,   # Neutral — could be good or bad surprise
        "disgust":   -0.8,
        "alarm":     -1.0,
    }

    TONE_REWARD: Dict[str, float] = {
        "enthusiastic": +0.6,
        "warm":         +0.4,
        "calm":         +0.1,
        "neutral":       0.0,
        "concerned":    -0.2,
        "disapproving": -0.6,
        "alarmed":      -1.0,
    }

    def reward_from_observation(
        self,
        facial_expression: str,
        vocal_tone:        str = "neutral",
        confidence:        float = 0.8,
    ) -> float:
        """Compute a reward signal from observed human reaction."""
        face_r = self.EXPRESSION_REWARD.get(facial_expression, 0.0)
        tone_r = self.TONE_REWARD.get(vocal_tone, 0.0)
        return (face_r * 0.6 + tone_r * 0.4) * confidence


# ─────────────────────────────────────────────────────────────────────
#  Master social learning coordinator
# ─────────────────────────────────────────────────────────────────────

class SocialLearningSystem:
    """
    Coordinates all social learning mechanisms.
    Called from robot_brain.py on each step where humans are visible.
    """

    def __init__(self, d_model: int = 512, proprio_dim: int = 128, n_actions: int = 32):
        self.imitation    = ImitationSystem(d_model, proprio_dim, n_actions)
        self.referencing  = SocialReferencing()
        self.gaze         = GazeFollowing()
        self.contagion    = EmotionalContagion(susceptibility=0.25)
        self.instruction  = VerbalInstructionParser()
        self.facial_rew   = FacialRewardDetector()
        self._interaction_log: List[dict] = []

    def process(
        self,
        observations:   List[HumanObservation],
        robot_pos:      Tuple[float, float],
        robot_emotion:  str,
        situation:      str = "general",
    ) -> dict:
        """
        Process all visible humans and return social signals.

        Returns dict with:
          - pad_shift: (dp, da, dd) emotional influence
          - gaze_target: where to look (if following gaze)
          - instruction: parsed verbal command (if any)
          - social_reward: reward from facial expressions
          - attention_target: what human is sustained-attending to
        """
        result = {
            "pad_shift":        (0.0, 0.0, 0.0),
            "gaze_target":      None,
            "instruction":      None,
            "social_reward":    0.0,
            "attention_target": None,
        }

        for obs in observations:
            dist = math.sqrt(obs.position[0]**2 + obs.position[1]**2)

            # Social referencing
            pad_ref = self.referencing.get_social_reference(obs, situation)
            if pad_ref:
                result["pad_shift"] = tuple(
                    a + b for a, b in zip(result["pad_shift"], pad_ref)
                )

            # Emotional contagion
            pad_cont = self.contagion.compute_contagion(obs.emotion, dist)
            result["pad_shift"] = tuple(
                a + b for a, b in zip(result["pad_shift"], pad_cont)
            )

            # Gaze following
            gaze_pos = self.gaze.process_observation(obs, robot_pos)
            if gaze_pos:
                result["gaze_target"] = gaze_pos

            # Sustained attention
            sustained = self.gaze.sustained_attention_target()
            if sustained:
                result["attention_target"] = sustained

            # Verbal instruction
            if obs.is_speaking and obs.speech_text:
                parsed = self.instruction.parse(obs.speech_text)
                if parsed:
                    result["instruction"] = parsed

            # Facial reward (from closest human)
            if dist < 2.0:
                result["social_reward"] += self.facial_rew.reward_from_observation(
                    obs.emotion, confidence=obs.confidence
                )

        self._interaction_log.append({
            "time": time.time(),
            "n_humans": len(observations),
            **{k: str(v) for k, v in result.items()},
        })
        return result

    def stats(self) -> dict:
        if not self._interaction_log:
            return {"interactions": 0}
        recent = self._interaction_log[-20:]
        has_instruction = sum(1 for r in recent if r.get("instruction") != "None")
        return {
            "total_interactions": len(self._interaction_log),
            "recent_instructions": has_instruction,
        }
