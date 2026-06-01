"""
ChimeraRobot Reward System
----------------------------
What makes the robot WANT things. Without reward signals, a robot
has no motivation to do anything at all.

Seven categories of reward (mirroring biological drives):

  1. Pain Avoidance      — negative reward from damage signals
                           (force overload, heat, collision)
  2. Homeostasis         — reward for maintaining optimal internal state
                           (battery level, temperature, joint health)
  3. Social Reward       — positive reward from human approval
                           (praise, cooperation, being understood)
  4. Curiosity           — intrinsic reward for novel states
                           (from curiosity.py, combined here)
  5. Task Completion     — extrinsic reward from achieving goals
                           (pick up object, reach waypoint)
  6. Competence          — reward for improving at skills over time
                           (a skill that used to fail now succeeds)
  7. Mastery             — long-term reward for becoming more capable
                           (tracks improvement over days/weeks)

These combine into a scalar reward signal that drives learning.
The weighting of each category reflects the robot's "values" —
changing these weights changes the robot's personality and priorities.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  Individual reward signals
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RewardSignal:
    name:        str
    value:       float          # Signed reward: positive = good, negative = bad
    category:    str            # Which of the 7 categories
    description: str = ""
    timestamp:   float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────
#  1. Pain Avoidance Rewards
# ─────────────────────────────────────────────────────────────────────

class PainRewardSystem:
    """
    Computes reward/penalty from physical damage signals.
    High pain = large negative reward = strong learning signal.
    This is the MOST IMPORTANT signal for robot safety.

    Sources:
      - Touch force exceeds threshold → pain
      - Temperature out of range → pain  
      - Joint torque overload → pain
      - Collision detected → pain
      - Battery critically low → pain (different quality)
    """

    def __init__(
        self,
        force_threshold: float = 0.7,    # [0,1] normalised
        temp_lo: float = 5.0,            # Celsius
        temp_hi: float = 50.0,
        battery_critical: float = 0.1,
        pain_scale: float = 2.0,         # Pain is weighted heavily
    ):
        self.force_threshold  = force_threshold
        self.temp_lo          = temp_lo
        self.temp_hi          = temp_hi
        self.battery_critical = battery_critical
        self.pain_scale       = pain_scale
        self._accumulated_pain = 0.0     # Running total (for health monitoring)

    def compute(
        self,
        touch_force:  float = 0.0,
        temperature:  float = 22.0,
        joint_torque: float = 0.0,       # Normalised [0,1]
        collision:    bool  = False,
        battery:      float = 1.0,
    ) -> List[RewardSignal]:
        signals = []

        # Force pain
        if touch_force > self.force_threshold:
            pain = -(touch_force - self.force_threshold) / (1.0 - self.force_threshold)
            signals.append(RewardSignal(
                "force_pain", pain * self.pain_scale, "pain",
                f"Force {touch_force:.2f} exceeds threshold {self.force_threshold}"
            ))

        # Temperature pain
        if temperature > self.temp_hi:
            pain = -(temperature - self.temp_hi) / 50.0
            signals.append(RewardSignal("heat_pain", pain * self.pain_scale, "pain",
                                        f"Overheating: {temperature:.1f}°C"))
        elif temperature < self.temp_lo:
            pain = -(self.temp_lo - temperature) / self.temp_lo
            signals.append(RewardSignal("cold_pain", pain * self.pain_scale, "pain",
                                        f"Too cold: {temperature:.1f}°C"))

        # Torque pain
        if joint_torque > 0.85:
            pain = -(joint_torque - 0.85) / 0.15
            signals.append(RewardSignal("torque_pain", pain * self.pain_scale, "pain",
                                        "Joint torque limit approached"))

        # Collision penalty
        if collision:
            signals.append(RewardSignal("collision", -0.5 * self.pain_scale, "pain",
                                        "Physical collision detected"))

        # Battery critical
        if battery < self.battery_critical:
            urgency = (self.battery_critical - battery) / self.battery_critical
            signals.append(RewardSignal("battery_critical", -urgency, "homeostasis",
                                        f"Battery at {battery*100:.1f}%"))

        total_pain = sum(abs(s.value) for s in signals if s.value < 0)
        self._accumulated_pain += total_pain
        return signals

    def cumulative_pain(self) -> float:
        return self._accumulated_pain

    def reset_accumulated(self):
        self._accumulated_pain = 0.0


# ─────────────────────────────────────────────────────────────────────
#  2. Homeostasis Rewards
# ─────────────────────────────────────────────────────────────────────

class HomeostasisReward:
    """
    Rewards the robot for maintaining a healthy internal state.
    Like hunger/thirst in animals — pushes toward optimal conditions.

    Homeostatic drives:
      - Battery: reward for keeping battery above 30%
      - Thermal: reward for keeping motors cool
      - Posture: reward for balanced, upright stance
      - Energy: reward for efficient movement (penalise waste)
    """

    def compute(
        self,
        battery:      float,    # [0, 1]
        motor_temp:   float,    # Celsius
        balance_error: float,  # [0, 1] — how tilted is the robot?
        energy_used:  float,   # Joules per step
        energy_budget: float = 50.0,
    ) -> List[RewardSignal]:
        signals = []

        # Battery homeostasis: target 50–100%
        if battery > 0.3:
            bat_reward = 0.1 * (battery - 0.3)
            signals.append(RewardSignal("battery_ok", bat_reward, "homeostasis"))
        
        # Thermal homeostasis: penalise high motor temperature
        if motor_temp > 60.0:
            signals.append(RewardSignal("motor_heat", -(motor_temp - 60.0) / 40.0,
                                        "homeostasis", "Motor running hot"))
        
        # Balance: penalise tilting over
        if balance_error > 0.3:
            signals.append(RewardSignal("balance_error", -balance_error * 0.5,
                                        "homeostasis", f"Balance error: {balance_error:.2f}"))
        else:
            signals.append(RewardSignal("balanced", 0.05, "homeostasis"))
        
        # Energy efficiency: penalise excess energy use
        if energy_used > energy_budget:
            efficiency_penalty = -(energy_used - energy_budget) / energy_budget * 0.1
            signals.append(RewardSignal("inefficient", efficiency_penalty, "homeostasis"))

        return signals


# ─────────────────────────────────────────────────────────────────────
#  3. Social Rewards
# ─────────────────────────────────────────────────────────────────────

class SocialRewardSystem:
    """
    Rewards derived from human interaction.
    Social creatures (like humans and many robots should be) derive
    deep satisfaction from positive social interactions.

    Signals:
      - Human expressed approval → positive reward
      - Human expressed displeasure → negative reward
      - Successful communication → positive reward
      - Human engaged attention → small positive reward
      - Helping a human complete a task → large positive reward
    """

    def __init__(self):
        self._interaction_count = 0
        self._approval_count    = 0
        self._disapproval_count = 0

    def human_rating(self, rating: float) -> RewardSignal:
        """
        Direct human feedback: +1 = excellent, 0 = neutral, -1 = very bad.
        This is the most valuable learning signal of all.
        """
        self._interaction_count += 1
        if rating > 0:
            self._approval_count += 1
        elif rating < 0:
            self._disapproval_count += 1

        # Scale up: human feedback is 3× more valuable than other signals
        return RewardSignal(
            "human_feedback", rating * 3.0, "social",
            f"Human rating: {rating:+.1f}"
        )

    def communication_success(self, understood: bool) -> RewardSignal:
        """Was the robot's message understood by the human?"""
        return RewardSignal(
            "communication", 0.3 if understood else -0.1, "social",
            "Message understood" if understood else "Communication failed"
        )

    def person_proximity(self, distance_m: float, target_m: float = 1.5) -> RewardSignal:
        """Small reward for being at a comfortable social distance from humans."""
        error = abs(distance_m - target_m)
        reward = 0.05 * math.exp(-error)
        return RewardSignal("social_proximity", reward, "social")

    def task_helped_human(self, task_name: str) -> RewardSignal:
        """Large reward for successfully helping a human accomplish something."""
        return RewardSignal(
            "helped_human", 5.0, "social",
            f"Successfully helped with: {task_name}"
        )

    def approval_rate(self) -> float:
        if self._interaction_count == 0:
            return 0.0
        return self._approval_count / self._interaction_count


# ─────────────────────────────────────────────────────────────────────
#  4. Task Completion Rewards
# ─────────────────────────────────────────────────────────────────────

class TaskRewardSystem:
    """
    Extrinsic rewards from accomplishing defined tasks.
    Tasks can be:
      - Explicit (set by human operator)
      - Inferred (derived from context)
      - Self-generated (robot sets its own goals)
    """

    TASK_REWARDS = {
        "pick_up_object":   3.0,
        "put_down_object":  2.0,
        "navigate_to_goal": 4.0,
        "open_door":        5.0,
        "wave_to_person":   2.0,
        "hand_object":      3.5,
        "avoid_obstacle":   1.0,
        "find_object":      2.5,
        "sit_down":         1.5,
        "stand_up":         1.5,
        "follow_person":    3.0,
    }

    def __init__(self):
        self._completed: List[Tuple[str, float, float]] = []  # (task, reward, time)
        self._failed:    List[str]                       = []

    def task_complete(self, task_name: str, quality: float = 1.0) -> RewardSignal:
        """
        Reward for completing a task.
        quality: 0–1 multiplier (did they do it well?)
        """
        base = self.TASK_REWARDS.get(task_name, 2.0)
        reward = base * quality
        self._completed.append((task_name, reward, time.time()))
        return RewardSignal(
            f"task_{task_name}", reward, "task",
            f"Completed: {task_name} (quality={quality:.2f})"
        )

    def task_failed(self, task_name: str) -> RewardSignal:
        self._failed.append(task_name)
        return RewardSignal(f"fail_{task_name}", -1.0, "task",
                            f"Failed: {task_name}")

    def subtask_progress(self, progress: float) -> RewardSignal:
        """Small reward for making progress toward a task."""
        return RewardSignal("progress", progress * 0.1, "task")

    def completion_rate(self) -> float:
        total = len(self._completed) + len(self._failed)
        return len(self._completed) / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────
#  5. Competence Reward (skill improvement)
# ─────────────────────────────────────────────────────────────────────

class CompetenceReward:
    """
    Rewards the robot for getting BETTER at things over time.
    This is different from task reward — you get competence reward
    even from partial success, if you improved vs last time.

    Tracks success rate per skill. When it improves → reward.
    When it gets worse → signal to practice more.

    This creates intrinsic motivation to practice — like a human
    feeling good when they improve at a skill they're working on.
    """

    def __init__(self):
        self._skill_history: Dict[str, List[bool]] = {}

    def record_attempt(self, skill_name: str, success: bool) -> Optional[RewardSignal]:
        if skill_name not in self._skill_history:
            self._skill_history[skill_name] = []
        history = self._skill_history[skill_name]
        history.append(success)

        # Need at least 5 attempts to compute improvement
        if len(history) < 5:
            return None

        recent  = sum(history[-5:])  / 5.0
        earlier = sum(history[-10:-5]) / 5.0 if len(history) >= 10 else 0.5

        improvement = recent - earlier
        if abs(improvement) < 0.05:
            return None   # Not enough change to signal

        reward = improvement * 2.0   # +2 for large improvement, -2 for large decline
        return RewardSignal(
            f"competence_{skill_name}",
            reward, "competence",
            f"{skill_name}: {earlier:.1%} → {recent:.1%} ({improvement:+.1%})"
        )

    def mastery_level(self, skill_name: str) -> float:
        history = self._skill_history.get(skill_name, [])
        if len(history) < 5:
            return 0.0
        return sum(history[-10:]) / min(len(history), 10)


# ─────────────────────────────────────────────────────────────────────
#  Combined reward aggregator
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    """Controls how much each reward category matters."""
    pain:        float = 3.0    # Pain is weighted highest (safety!)
    homeostasis: float = 0.5
    social:      float = 2.0    # Social approval is very important
    curiosity:   float = 0.8
    task:        float = 1.5
    competence:  float = 0.6
    mastery:     float = 0.3


@dataclass
class AggregatedReward:
    total:      float
    breakdown:  Dict[str, float]
    signals:    List[RewardSignal]
    dominant:   str         # Which category contributed most


class RewardAggregator:
    """
    Combines all reward signals into a single scalar with breakdown.
    This is the signal that drives all learning in the robot.
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.weights    = weights or RewardWeights()
        self.pain_sys   = PainRewardSystem()
        self.homeo_sys  = HomeostasisReward()
        self.social_sys = SocialRewardSystem()
        self.task_sys   = TaskRewardSystem()
        self.comp_sys   = CompetenceReward()
        self._history: List[AggregatedReward] = []

    def step_reward(
        self,
        # Pain inputs
        touch_force:    float = 0.0,
        temperature:    float = 22.0,
        collision:      bool  = False,
        # Homeostasis inputs
        battery:        float = 1.0,
        balance_error:  float = 0.0,
        energy_used:    float = 0.0,
        # Curiosity (from curiosity.py)
        curiosity_reward: float = 0.0,
        # Task
        task_events:    Optional[List[str]] = None,
    ) -> AggregatedReward:
        """Compute the full reward signal for one timestep."""
        all_signals: List[RewardSignal] = []
        breakdown:   Dict[str, float]   = {}

        # Pain
        pain_signals = self.pain_sys.compute(touch_force, temperature, 0.0, collision, battery)
        all_signals.extend(pain_signals)

        # Homeostasis
        homeo_signals = self.homeo_sys.compute(battery, temperature, balance_error, energy_used)
        all_signals.extend(homeo_signals)

        # Curiosity (passed in from CuriosityDrive)
        if curiosity_reward != 0.0:
            all_signals.append(RewardSignal("curiosity", curiosity_reward, "curiosity"))

        # Task events
        for task_name in (task_events or []):
            all_signals.append(self.task_sys.task_complete(task_name))

        # Aggregate by category with weights
        weight_map = {
            "pain":        self.weights.pain,
            "homeostasis": self.weights.homeostasis,
            "social":      self.weights.social,
            "curiosity":   self.weights.curiosity,
            "task":        self.weights.task,
            "competence":  self.weights.competence,
        }
        for cat, w in weight_map.items():
            cat_sum = sum(s.value for s in all_signals if s.category == cat)
            breakdown[cat] = round(cat_sum * w, 4)

        total = sum(breakdown.values())
        dominant = max(breakdown, key=lambda k: abs(breakdown[k])) if breakdown else "none"

        agg = AggregatedReward(
            total=round(total, 4),
            breakdown=breakdown,
            signals=all_signals,
            dominant=dominant,
        )
        self._history.append(agg)
        if len(self._history) > 1000:
            self._history = self._history[-1000:]
        return agg

    def add_social_signal(self, rating: float) -> RewardSignal:
        """Inject a human rating signal (called from chatbot)."""
        sig = self.social_sys.human_rating(rating)
        return sig

    def recent_average(self, window: int = 20) -> float:
        recent = self._history[-window:]
        return sum(r.total for r in recent) / len(recent) if recent else 0.0

    def stats(self) -> dict:
        return {
            "recent_avg_reward":  round(self.recent_average(), 4),
            "task_completion_%":  round(self.task_sys.completion_rate() * 100, 1),
            "social_approval_%":  round(self.social_sys.approval_rate() * 100, 1),
            "cumulative_pain":    round(self.pain_sys.cumulative_pain(), 3),
        }

    def describe(self, agg: AggregatedReward) -> str:
        lines = [f"  Total reward: {agg.total:+.4f}  [{agg.dominant}]"]
        for cat, val in agg.breakdown.items():
            if abs(val) > 0.001:
                bar = "+" * int(max(0,  val) * 10) + "-" * int(max(0, -val) * 10)
                lines.append(f"    {cat:<14} {val:+.4f}  {bar}")
        return "\n".join(lines)
