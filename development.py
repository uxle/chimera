"""
ChimeraRobot Developmental Stages
------------------------------------
This is what makes the robot evolve like a baby brain.

A newborn baby doesn't start knowing how to walk, talk, or understand
complex concepts. These capabilities emerge in stages, driven by:
  • Physical maturation (motor circuits becoming active)
  • Accumulated experience (millions of sensorimotor interactions)
  • Social scaffolding (caregivers guide learning)

ChimeraRobot mimics this developmental arc:

  Stage 0: NEWBORN    (0–1k steps)
    • Random motor movements (exploring body)
    • Basic pain/pleasure responses
    • No language, no goals
    • Just sensing and moving

  Stage 1: INFANT     (1k–10k steps)
    • Directed attention (curiosity active)
    • Object permanence emerging
    • Simple cause-and-effect learning
    • First words/responses

  Stage 2: TODDLER    (10k–50k steps)
    • Goal-directed reaching and grasping
    • Social referencing (looking at humans for cues)
    • Simple language understanding
    • Emotion expression reliable

  Stage 3: CHILD      (50k–200k steps)
    • Full motor coordination
    • Language comprehension and production
    • Multi-step planning
    • Empathy responses

  Stage 4: ADOLESCENT (200k+ steps)
    • Abstract reasoning
    • Self-model (theory of mind for self)
    • Complex social skills
    • Autonomous goal generation

Each stage UNLOCKS new capabilities and INCREASES the complexity
of the sensory processing, motor outputs, and language generation.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────
#  Developmental stages
# ─────────────────────────────────────────────────────────────────────

class Stage(IntEnum):
    NEWBORN    = 0
    INFANT     = 1
    TODDLER    = 2
    CHILD      = 3
    ADOLESCENT = 4

STAGE_NAMES = {
    Stage.NEWBORN:    "Newborn",
    Stage.INFANT:     "Infant",
    Stage.TODDLER:    "Toddler",
    Stage.CHILD:      "Child",
    Stage.ADOLESCENT: "Adolescent",
}

# Step thresholds for advancing to each stage
STAGE_THRESHOLDS = {
    Stage.NEWBORN:    0,
    Stage.INFANT:     1_000,
    Stage.TODDLER:    10_000,
    Stage.CHILD:      50_000,
    Stage.ADOLESCENT: 200_000,
}


# ─────────────────────────────────────────────────────────────────────
#  Capability flags per stage
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CapabilitySet:
    """What the robot can do at each developmental stage."""

    # Motor capabilities
    can_move_randomly:      bool = True    # Always
    can_directed_reach:     bool = False   # Toddler+
    can_walk:               bool = False   # Toddler+
    can_grasp_objects:      bool = False   # Toddler+
    can_coordinated_bimanual: bool = False # Child+
    can_fine_manipulation:  bool = False   # Adolescent+

    # Sensory capabilities
    uses_vision:            bool = False   # Infant+
    uses_audio:             bool = False   # Infant+
    uses_touch_feedback:    bool = True    # Always (pain/pleasure)
    uses_proprioception:    bool = False   # Toddler+
    full_multimodal_fusion: bool = False   # Child+

    # Cognitive capabilities
    has_object_permanence:  bool = False   # Infant+
    has_goal_directed_action: bool = False # Toddler+
    has_language:           bool = False   # Toddler+ (basic)
    has_planning:           bool = False   # Child+
    has_self_model:         bool = False   # Adolescent+
    has_empathy:            bool = False   # Child+

    # Emotional capabilities
    basic_emotions:         bool = True    # Always
    social_emotions:        bool = False   # Infant+ (fear of strangers, attachment)
    complex_emotions:       bool = False   # Child+ (pride, shame, guilt)
    emotional_regulation:   bool = False   # Adolescent+

    # Social capabilities
    social_referencing:     bool = False   # Infant+ (checking human reactions)
    turn_taking:            bool = False   # Toddler+
    cooperative_play:       bool = False   # Child+
    theory_of_mind:         bool = False   # Adolescent+

    # Learning capabilities
    learning_rate_multiplier: float = 1.0
    memory_consolidation:   bool = False   # Toddler+
    curiosity_active:       bool = False   # Infant+
    imitation_learning:     bool = False   # Infant+


# Define capabilities at each stage (cumulative — each stage adds to previous)
STAGE_CAPABILITIES: Dict[Stage, CapabilitySet] = {
    Stage.NEWBORN: CapabilitySet(
        uses_touch_feedback=True,
        basic_emotions=True,
        learning_rate_multiplier=2.0,     # Newborns learn FAST
    ),
    Stage.INFANT: CapabilitySet(
        uses_touch_feedback=True,
        uses_vision=True,
        uses_audio=True,
        has_object_permanence=True,
        basic_emotions=True,
        social_emotions=True,
        social_referencing=True,
        curiosity_active=True,
        imitation_learning=True,
        learning_rate_multiplier=1.8,
    ),
    Stage.TODDLER: CapabilitySet(
        uses_vision=True,
        uses_audio=True,
        uses_touch_feedback=True,
        uses_proprioception=True,
        has_object_permanence=True,
        has_goal_directed_action=True,
        has_language=True,
        can_directed_reach=True,
        can_walk=True,
        can_grasp_objects=True,
        memory_consolidation=True,
        basic_emotions=True,
        social_emotions=True,
        social_referencing=True,
        curiosity_active=True,
        imitation_learning=True,
        turn_taking=True,
        learning_rate_multiplier=1.5,
    ),
    Stage.CHILD: CapabilitySet(
        uses_vision=True,
        uses_audio=True,
        uses_touch_feedback=True,
        uses_proprioception=True,
        full_multimodal_fusion=True,
        has_object_permanence=True,
        has_goal_directed_action=True,
        has_language=True,
        has_planning=True,
        has_empathy=True,
        can_directed_reach=True,
        can_walk=True,
        can_grasp_objects=True,
        can_coordinated_bimanual=True,
        memory_consolidation=True,
        basic_emotions=True,
        social_emotions=True,
        complex_emotions=True,
        social_referencing=True,
        curiosity_active=True,
        imitation_learning=True,
        turn_taking=True,
        cooperative_play=True,
        emotional_regulation=False,     # Still developing
        learning_rate_multiplier=1.2,
    ),
    Stage.ADOLESCENT: CapabilitySet(
        uses_vision=True,
        uses_audio=True,
        uses_touch_feedback=True,
        uses_proprioception=True,
        full_multimodal_fusion=True,
        has_object_permanence=True,
        has_goal_directed_action=True,
        has_language=True,
        has_planning=True,
        has_self_model=True,
        has_empathy=True,
        can_directed_reach=True,
        can_walk=True,
        can_grasp_objects=True,
        can_coordinated_bimanual=True,
        can_fine_manipulation=True,
        memory_consolidation=True,
        basic_emotions=True,
        social_emotions=True,
        complex_emotions=True,
        emotional_regulation=True,
        social_referencing=True,
        curiosity_active=True,
        imitation_learning=True,
        turn_taking=True,
        cooperative_play=True,
        theory_of_mind=True,
        learning_rate_multiplier=1.0,
    ),
}


# ─────────────────────────────────────────────────────────────────────
#  Developmental milestones
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Milestone:
    name:        str
    description: str
    achieved_at: Optional[int] = None   # Step when achieved
    achieved:    bool = False

    def achieve(self, step: int):
        if not self.achieved:
            self.achieved    = True
            self.achieved_at = step
            print(f"  🏆 MILESTONE: {self.name} — {self.description}")


# All milestones the robot can achieve through experience
ALL_MILESTONES: List[Milestone] = [
    Milestone("first_pain",        "Experienced pain for the first time"),
    Milestone("first_joy",         "Experienced joy for the first time"),
    Milestone("first_object",      "Recognised an object visually"),
    Milestone("first_person",      "Recognised a human in the scene"),
    Milestone("first_word",        "Produced a meaningful word/response"),
    Milestone("first_grasp",       "Successfully grasped an object"),
    Milestone("first_step",        "Took a first walking step"),
    Milestone("first_task",        "Completed a defined task"),
    Milestone("first_wave",        "Waved to a human"),
    Milestone("fear_conditioned",  "Learned to fear a specific stimulus"),
    Milestone("learned_skill",     "Learned a new motor skill"),
    Milestone("spatial_map",       "Built a basic map of the environment"),
    Milestone("social_bond",       "Formed a positive association with a human"),
    Milestone("hundred_episodes",  "Completed 100 episodes"),
    Milestone("thousand_steps",    "Completed 1000 brain steps"),
    Milestone("self_improvement",  "Completed first self-improvement cycle"),
    Milestone("first_question",    "Asked a question to a human"),
    Milestone("task_master",       "Achieved >80% task completion rate"),
    Milestone("emotionally_stable","Maintained neutral emotion for 100 steps"),
    Milestone("explorer",          "Mapped >50% of environment"),
]


# ─────────────────────────────────────────────────────────────────────
#  Developmental system
# ─────────────────────────────────────────────────────────────────────

class DevelopmentalSystem:
    """
    Tracks the robot's developmental stage and unlocks capabilities
    as the robot gains experience.

    This is the heart of the "baby brain that evolves" concept:
    the robot genuinely cannot do things it hasn't developmentally
    reached yet — and as it gains experience, new capabilities emerge.

    The robot's "age" is measured in experience steps, not clock time.
    """

    def __init__(self, save_path: str = "artifacts/development.json"):
        self.save_path   = save_path
        self._step_count = 0
        self._stage      = Stage.NEWBORN
        self._milestones = list(ALL_MILESTONES)
        self._capability = STAGE_CAPABILITIES[Stage.NEWBORN]
        self._stage_history: List[Tuple[Stage, int, float]] = []

        # Development metrics for advancement
        self._metrics: Dict[str, float] = {
            "avg_reward":       0.0,
            "task_success_rate": 0.0,
            "language_uses":    0.0,
            "pain_events":      0.0,
            "joy_events":       0.0,
            "objects_recognised": 0.0,
        }
        self._load()

    # ── Stage management ─────────────────────────────────────────────

    @property
    def stage(self) -> Stage:
        return self._stage

    @property
    def capabilities(self) -> CapabilitySet:
        return self._capability

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self._stage]

    def step(self, step_count: int, metrics: Optional[Dict[str, float]] = None):
        """
        Called every N steps to check for stage advancement.
        Updates metrics and possibly advances the stage.
        """
        self._step_count = step_count
        if metrics:
            for k, v in metrics.items():
                if k in self._metrics:
                    # Exponential moving average
                    self._metrics[k] = self._metrics[k] * 0.99 + v * 0.01

        new_stage = self._compute_stage()
        if new_stage > self._stage:
            self._advance_to(new_stage)

    def _compute_stage(self) -> Stage:
        """
        Stage is determined by experience count AND performance metrics.
        Just having many steps isn't enough — the robot must demonstrate
        the capabilities of each stage.
        """
        for stage in reversed(list(Stage)):
            threshold = STAGE_THRESHOLDS[stage]
            if self._step_count >= threshold:
                # Additional performance check for higher stages
                if stage >= Stage.TODDLER:
                    if self._metrics["avg_reward"] < -0.5:
                        continue   # Not performing well enough
                if stage >= Stage.CHILD:
                    if self._metrics["task_success_rate"] < 0.3:
                        continue
                return stage
        return Stage.NEWBORN

    def _advance_to(self, new_stage: Stage):
        """Advance to a new developmental stage."""
        old = self._stage
        self._stage      = new_stage
        self._capability = STAGE_CAPABILITIES[new_stage]
        self._stage_history.append((new_stage, self._step_count, time.time()))
        self._save()

        print(f"\n  {'='*60}")
        print(f"  🌟 DEVELOPMENTAL MILESTONE: {STAGE_NAMES[old]} → {STAGE_NAMES[new_stage]}")
        print(f"  Step: {self._step_count:,}")
        print(f"  New capabilities unlocked:")
        old_cap = STAGE_CAPABILITIES[old]
        new_cap = self._capability
        for attr in vars(new_cap):
            if attr == "learning_rate_multiplier":
                continue
            old_val = getattr(old_cap, attr)
            new_val = getattr(new_cap, attr)
            if new_val and not old_val:
                print(f"    ✅ {attr.replace('_', ' ').title()}")
        print(f"  {'='*60}\n")

    # ── Milestone tracking ───────────────────────────────────────────

    def check_milestone(self, name: str):
        """Mark a milestone as achieved."""
        for m in self._milestones:
            if m.name == name and not m.achieved:
                m.achieve(self._step_count)
                self._save()
                break

    def achieved_milestones(self) -> List[Milestone]:
        return [m for m in self._milestones if m.achieved]

    def pending_milestones(self) -> List[Milestone]:
        return [m for m in self._milestones if not m.achieved]

    # ── Capability queries ───────────────────────────────────────────

    def can(self, capability: str) -> bool:
        """Quick check: does the robot have this capability now?"""
        return bool(getattr(self._capability, capability, False))

    def learning_rate_multiplier(self) -> float:
        return self._capability.learning_rate_multiplier

    # ── Newborn behaviour: random motor exploration ──────────────────

    def newborn_motor_noise(self, d_action: int = 32) -> torch.Tensor:
        """
        Newborns move randomly to explore their body.
        This is the robot equivalent of a baby kicking and waving arms.
        The amplitude decreases as the robot matures.
        """
        # Noise amplitude decreases with stage
        amplitude = max(0.05, 1.0 - self._stage * 0.25)
        return torch.randn(d_action) * amplitude

    def filter_action(
        self,
        intended_action: torch.Tensor,
        d_action: int = 32,
    ) -> torch.Tensor:
        """
        Apply developmental constraints to an action:
        - Newborn: mostly noise, little control
        - Infant: noisy but directional
        - Toddler+: mostly controlled with some motor noise
        """
        noise_level = max(0.0, 1.0 - self._stage * 0.3)
        noise = torch.randn_like(intended_action) * noise_level
        control_weight = 1.0 - noise_level
        return intended_action * control_weight + noise

    # ── Attention gating ────────────────────────────────────────────

    def attention_gate(self, modalities: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
        """
        Gate which sensory modalities are processed based on developmental stage.
        Newborns can't really use vision or audio yet — they're just noise.
        This forces the robot to learn to use its senses progressively.
        """
        cap = self._capability
        gated = {}
        for name, emb in modalities.items():
            if name == "vision"  and not cap.uses_vision:         emb = None
            if name == "audio"   and not cap.uses_audio:          emb = None
            if name == "proprio" and not cap.uses_proprioception:  emb = None
            gated[name] = emb
        return gated

    # ── Language gating ─────────────────────────────────────────────

    def max_response_length(self) -> int:
        """
        Language complexity grows with stage.
        Newborn: no language
        Infant: single words
        Toddler: simple sentences
        Child: paragraphs
        Adolescent: full discourse
        """
        lengths = {
            Stage.NEWBORN:    0,
            Stage.INFANT:     5,    # tokens
            Stage.TODDLER:    30,
            Stage.CHILD:      100,
            Stage.ADOLESCENT: 512,
        }
        return lengths[self._stage]

    # ── Progress report ──────────────────────────────────────────────

    def progress_to_next_stage(self) -> float:
        """How far (0–1) toward the next developmental stage."""
        if self._stage == Stage.ADOLESCENT:
            return 1.0
        next_stage    = Stage(self._stage + 1)
        current_thresh = STAGE_THRESHOLDS[self._stage]
        next_thresh   = STAGE_THRESHOLDS[next_stage]
        progress = (self._step_count - current_thresh) / (next_thresh - current_thresh)
        return max(0.0, min(1.0, progress))

    def report(self) -> str:
        cap = self._capability
        lines = [
            f"  Developmental Stage : {self.stage_name} (stage {self._stage})",
            f"  Experience steps    : {self._step_count:,}",
            f"  Progress to next    : {self.progress_to_next_stage()*100:.1f}%",
            f"  Learning rate       : ×{cap.learning_rate_multiplier:.1f}",
            f"",
            f"  Active capabilities :",
        ]
        for attr, val in vars(cap).__dict__.items() if hasattr(cap, '__dict__') else asdict(cap).items():
            if val and attr != "learning_rate_multiplier":
                lines.append(f"    ✅ {attr.replace('_', ' ').title()}")
        lines.append(f"")
        lines.append(f"  Milestones: {len(self.achieved_milestones())}/{len(self._milestones)} achieved")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        data = {
            "step_count":    self._step_count,
            "stage":         int(self._stage),
            "metrics":       self._metrics,
            "stage_history": [(int(s), c, t) for s, c, t in self._stage_history],
            "milestones":    {m.name: {"achieved": m.achieved, "at": m.achieved_at}
                              for m in self._milestones},
        }
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        if not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path) as f:
                data = json.load(f)
            self._step_count = data.get("step_count", 0)
            self._stage      = Stage(data.get("stage", 0))
            self._capability = STAGE_CAPABILITIES[self._stage]
            self._metrics    = data.get("metrics", self._metrics)
            for m in self._milestones:
                md = data.get("milestones", {}).get(m.name, {})
                m.achieved    = md.get("achieved", False)
                m.achieved_at = md.get("at", None)
            print(f"[Development] Loaded: {self.stage_name} | step {self._step_count:,}")
        except Exception as e:
            print(f"[Development] Could not load state: {e}")
