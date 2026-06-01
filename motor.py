"""
ChimeraRobot Motor Control
---------------------------
Translates brain decisions into physical robot movements.

Systems:
  HandController  — 5-finger grasping, pointing, waving, releasing
  LegController   — bipedal walking, turning, balance, sitting
  HeadController  — look direction, nodding, shaking
  BodyController  — coordinates all limbs for complex motions

All controllers:
  • Respect joint limits (won't break the robot)
  • Scale speed/force by emotional state (fear → cautious, joy → energetic)
  • Provide safety stops (stop if pain signal is high)
  • Work with real serial/USB hardware OR simulator
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

from robot_config import HandConfig, LegConfig


# ─────────────────────────────────────────────────────────────────────
#  Joint state
# ─────────────────────────────────────────────────────────────────────

@dataclass
class JointState:
    name:        str
    position:    float    # Radians (or metres for prismatic)
    velocity:    float    # Rad/s
    torque:      float    # Nm
    min_pos:     float = -math.pi
    max_pos:     float =  math.pi
    max_torque:  float = 5.0

    def clamp(self, target: float) -> float:
        return max(self.min_pos, min(self.max_pos, target))

    def is_at_limit(self) -> bool:
        margin = 0.05  # 3 degrees
        return (self.position < self.min_pos + margin or
                self.position > self.max_pos - margin)


# ─────────────────────────────────────────────────────────────────────
#  Named gestures and gaits
# ─────────────────────────────────────────────────────────────────────

class HandGesture(Enum):
    OPEN      = auto()   # All fingers extended
    FIST      = auto()   # All fingers closed
    POINT     = auto()   # Index extended, others closed
    PINCH     = auto()   # Thumb + index
    WAVE      = auto()   # Open hand, wave side to side
    THUMBS_UP = auto()   # Fist + thumb extended
    PEACE     = auto()   # Index + middle extended
    GRASP     = auto()   # Wrap around object
    GENTLE    = auto()   # Partial close, low force


class GaitMode(Enum):
    STAND        = auto()
    WALK_FORWARD = auto()
    WALK_BACK    = auto()
    TURN_LEFT    = auto()
    TURN_RIGHT   = auto()
    SIT          = auto()
    CROUCH       = auto()
    STEP_OVER    = auto()
    RUN          = auto()


# Finger joint targets per gesture (5 fingers × 3 joints = 15 values)
# Values in [0, 1] where 0 = fully open, 1 = fully closed
GESTURE_TARGETS: Dict[HandGesture, List[float]] = {
    HandGesture.OPEN:      [0]*15,
    HandGesture.FIST:      [1]*15,
    HandGesture.POINT:     [1,1,1, 0,0,0, 1,1,1, 1,1,1, 1,1,1],  # index open
    HandGesture.PINCH:     [0,0.7,1, 0,0.7,1, 1,1,1, 1,1,1, 1,1,1],
    HandGesture.THUMBS_UP: [0,0,0, 1,1,1, 1,1,1, 1,1,1, 1,1,1],
    HandGesture.PEACE:     [1,1,1, 0,0,0, 0,0,0, 1,1,1, 1,1,1],
    HandGesture.GRASP:     [0.8]*15,
    HandGesture.GENTLE:    [0.4]*15,
    HandGesture.WAVE:      [0]*15,   # Wave is an animation, not just joint pos
}


# ─────────────────────────────────────────────────────────────────────
#  Motor command structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HandCommand:
    gesture:    HandGesture
    force:      float = 0.5           # 0–1 grip force
    speed:      float = 0.5           # 0–1 movement speed
    wrist_pan:  float = 0.0           # Radians (-pi/2 to pi/2)
    wrist_tilt: float = 0.0           # Radians
    arm_x:      float = 0.0           # Target end-effector position (m)
    arm_y:      float = 0.0
    arm_z:      float = 0.5           # Height from ground

@dataclass
class LegCommand:
    gait:           GaitMode
    speed:          float = 0.3       # m/s
    step_height:    float = 0.06      # m
    turn_rate:      float = 0.0       # rad/s (for turn gaits)
    balance_adjust: float = 0.0       # Lateral CoM adjustment

@dataclass
class HeadCommand:
    pan:    float = 0.0               # Yaw: negative = left, positive = right (rad)
    tilt:   float = 0.0              # Pitch: negative = down, positive = up (rad)
    speed:  float = 0.5

@dataclass
class BodyCommand:
    left_hand:  Optional[HandCommand] = None
    right_hand: Optional[HandCommand] = None
    legs:       Optional[LegCommand]  = None
    head:       Optional[HeadCommand] = None
    timestamp:  float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────────
#  Hand controller
# ─────────────────────────────────────────────────────────────────────

class HandController:
    """Controls one robotic hand."""

    def __init__(self, config: HandConfig, side: str = "right"):
        self.config = config
        self.side   = side
        self.joints: List[JointState] = self._init_joints()
        self._current_gesture = HandGesture.OPEN
        self._wave_phase = 0.0

    def _init_joints(self) -> List[JointState]:
        joints = []
        for finger in range(self.config.n_fingers):
            for joint in range(self.config.n_joints_per_finger):
                joints.append(JointState(
                    name=f"finger_{finger}_joint_{joint}",
                    position=0.0, velocity=0.0, torque=0.0,
                    min_pos=-0.1, max_pos=math.pi * 0.8,
                    max_torque=self.config.max_torque_nm,
                ))
        for w in range(self.config.n_wrist_joints):
            joints.append(JointState(
                name=f"wrist_{w}",
                position=0.0, velocity=0.0, torque=0.0,
                min_pos=-math.pi/2, max_pos=math.pi/2,
                max_torque=self.config.max_torque_nm * 2,
            ))
        return joints

    def execute(self, cmd: HandCommand, emotion_weights: Dict[str, float],
                pain_level: float = 0.0) -> List[float]:
        """
        Execute a hand command and return target joint positions.
        Scales force/speed by emotional state.

        Args:
            cmd:              HandCommand
            emotion_weights:  from EmotionSystem.behavioral_weights()
            pain_level:       if > 0.5, reduce force and speed

        Returns:
            List of target joint positions (radians)
        """
        # Safety: if in pain, go gentle
        if pain_level > 0.5:
            cmd = HandCommand(
                gesture=HandGesture.GENTLE,
                force=max(0.1, cmd.force * (1.0 - pain_level)),
                speed=max(0.1, cmd.speed * 0.3),
            )

        # Scale by emotion
        fear_level  = emotion_weights.get("avoid", 0.0)
        joy_level   = emotion_weights.get("approach", 0.0)
        speed_scale = max(0.1, min(1.5, 1.0 + joy_level * 0.3 - fear_level * 0.3))
        force_scale = max(0.1, min(1.0, cmd.force * (1.0 - fear_level * 0.5)))

        targets = self._compute_finger_targets(cmd.gesture, force_scale)

        # Wrist
        wrist_pan  = max(-math.pi/2, min(math.pi/2, cmd.wrist_pan))
        wrist_tilt = max(-math.pi/2, min(math.pi/2, cmd.wrist_tilt))
        targets.extend([wrist_pan, wrist_tilt])

        self._current_gesture = cmd.gesture
        return targets

    def _compute_finger_targets(self, gesture: HandGesture, force: float) -> List[float]:
        """Compute actual radian targets from gesture template."""
        template = GESTURE_TARGETS.get(gesture, GESTURE_TARGETS[HandGesture.OPEN])

        # Wave animation: oscillate index and middle fingers
        if gesture == HandGesture.WAVE:
            self._wave_phase += 0.3
            template = template.copy()
            template[3] = 0.4 + 0.4 * math.sin(self._wave_phase)
            template[6] = 0.4 + 0.4 * math.sin(self._wave_phase + 0.5)

        # Map [0,1] template to [0, max_bend] radians
        max_bend = math.pi * 0.75
        return [t * max_bend * force for t in template]

    def describe(self) -> str:
        return f"{self.side} hand: {self._current_gesture.name}"


# ─────────────────────────────────────────────────────────────────────
#  Leg controller (bipedal gait generator)
# ─────────────────────────────────────────────────────────────────────

class LegController:
    """
    Generates walking gaits for bipedal locomotion.
    Uses a Central Pattern Generator (CPG) — an oscillator that creates
    rhythmic leg movements without needing full trajectory planning.

    The CPG frequency adapts to speed command.
    Balance is maintained by shifting the centre of mass.
    """

    def __init__(self, config: LegConfig):
        self.config = config
        self._phase = 0.0             # CPG phase [0, 2π]
        self._gait  = GaitMode.STAND
        self._step_count = 0

        # Joint states: hip_x, hip_y, hip_z, knee, ankle_x, ankle_z × 2 legs
        self.joints: Dict[str, JointState] = {
            f"{side}_{j}": JointState(
                name=f"{side}_{j}", position=0.0, velocity=0.0, torque=0.0,
                min_pos=lo, max_pos=hi, max_torque=config.max_torque_nm,
            )
            for side in ["left", "right"]
            for j, lo, hi in [
                ("hip_x",   -0.4, 0.4),
                ("hip_y",   -0.4, 0.5),
                ("hip_z",   -0.2, 0.2),
                ("knee",     0.0, 1.8),
                ("ankle_x", -0.5, 0.5),
                ("ankle_z", -0.2, 0.2),
            ]
        }

    def step(self, cmd: LegCommand, emotion_weights: Dict[str, float],
             dt: float = 0.01) -> Dict[str, float]:
        """
        Advance the gait by one timestep.
        Returns target joint positions for all leg joints.

        Emotion effects:
          fear → slow down, shorten steps
          joy  → can run, larger steps
          sadness → slow, shuffling gait
        """
        fear = emotion_weights.get("avoid", 0.0)
        joy  = emotion_weights.get("approach", 0.0)

        speed = cmd.speed * (1.0 + joy * 0.4 - fear * 0.5)
        speed = max(0.0, min(self.config.max_speed_ms, speed))
        step_h = cmd.step_height * (1.0 - fear * 0.3)

        # Update CPG phase
        freq = speed / 0.5             # ~1 Hz at 0.5 m/s
        self._phase = (self._phase + 2 * math.pi * freq * dt) % (2 * math.pi)

        self._gait = cmd.gait
        targets: Dict[str, float] = {}

        if cmd.gait == GaitMode.STAND:
            # Standing still — all joints at neutral
            for name in self.joints:
                targets[name] = 0.0

        elif cmd.gait in (GaitMode.WALK_FORWARD, GaitMode.WALK_BACK):
            direction = 1.0 if cmd.gait == GaitMode.WALK_FORWARD else -1.0
            # Left and right legs are antiphase (offset by π)
            for side, phase_offset in [("left", 0.0), ("right", math.pi)]:
                ph = self._phase + phase_offset
                swing = max(0, math.sin(ph))       # 0 when foot is on ground

                hip_y    =  direction * 0.2 * math.sin(ph)
                knee     =  step_h * 4 * swing
                ankle_x  = -hip_y * 0.5
                lift     =  step_h * swing

                targets[f"{side}_hip_x"]   = 0.0
                targets[f"{side}_hip_y"]   = hip_y
                targets[f"{side}_hip_z"]   = cmd.balance_adjust
                targets[f"{side}_knee"]    = max(0, knee)
                targets[f"{side}_ankle_x"] = ankle_x
                targets[f"{side}_ankle_z"] = 0.0

        elif cmd.gait in (GaitMode.TURN_LEFT, GaitMode.TURN_RIGHT):
            rate = cmd.turn_rate * (1.0 if cmd.gait == GaitMode.TURN_RIGHT else -1.0)
            for side in ["left", "right"]:
                sign = 1.0 if side == "right" else -1.0
                targets[f"{side}_hip_x"]   = 0.0
                targets[f"{side}_hip_y"]   = rate * sign * 0.15
                targets[f"{side}_hip_z"]   = rate * 0.1
                targets[f"{side}_knee"]    = 0.1
                targets[f"{side}_ankle_x"] = 0.0
                targets[f"{side}_ankle_z"] = 0.0

        elif cmd.gait == GaitMode.SIT:
            for side in ["left", "right"]:
                targets[f"{side}_hip_x"]   = 0.0
                targets[f"{side}_hip_y"]   = -0.3
                targets[f"{side}_hip_z"]   = 0.0
                targets[f"{side}_knee"]    = 1.2
                targets[f"{side}_ankle_x"] = 0.3
                targets[f"{side}_ankle_z"] = 0.0

        # Clamp all targets to joint limits
        for name, val in targets.items():
            j = self.joints.get(name)
            if j:
                targets[name] = j.clamp(val)

        self._step_count += 1
        return targets

    def describe(self) -> str:
        return f"gait={self._gait.name} phase={self._phase:.2f}"


# ─────────────────────────────────────────────────────────────────────
#  Head controller
# ─────────────────────────────────────────────────────────────────────

class HeadController:
    """Controls head pan (yaw) and tilt (pitch). Adds attention behaviors."""

    def __init__(self):
        self.pan  = 0.0    # Current position (rad)
        self.tilt = 0.0
        self._nod_phase = 0.0
        self._behavior  = "neutral"

    def execute(self, cmd: HeadCommand, emotion_name: str = "NEUTRAL") -> Tuple[float, float]:
        """Returns (pan, tilt) in radians."""
        target_pan  = max(-1.5, min(1.5, cmd.pan))
        target_tilt = max(-0.5, min(0.5, cmd.tilt))
        speed       = max(0.05, min(1.0, cmd.speed))

        # Smooth movement toward target
        self.pan  += (target_pan  - self.pan)  * speed
        self.tilt += (target_tilt - self.tilt) * speed

        # Emotional behaviors
        if emotion_name == "FEAR":
            # Look toward threat (assume threat is in the direction we last moved)
            self.tilt = -0.1   # Slightly down (guarded)
        elif emotion_name == "SURPRISE":
            self.tilt = 0.2    # Look up/alert
        elif emotion_name == "SADNESS":
            self.tilt = -0.3   # Head down
        elif emotion_name == "JOY":
            # Gentle nod
            self._nod_phase += 0.2
            self.tilt += 0.05 * math.sin(self._nod_phase)

        return round(self.pan, 4), round(self.tilt, 4)


# ─────────────────────────────────────────────────────────────────────
#  Body coordinator
# ─────────────────────────────────────────────────────────────────────

class BodyController:
    """
    Top-level coordinator for all robot actuators.
    Takes a BodyCommand → distributes to sub-controllers → returns all joint targets.
    """

    def __init__(self, hand_cfg: HandConfig, leg_cfg: LegConfig):
        self.left_hand  = HandController(hand_cfg, "left")
        self.right_hand = HandController(hand_cfg, "right")
        self.legs       = LegController(leg_cfg)
        self.head       = HeadController()
        self._emergency_stop = False

    def emergency_stop(self):
        """Immediately command all joints to zero velocity."""
        self._emergency_stop = True

    def clear_stop(self):
        self._emergency_stop = False

    def execute(
        self,
        cmd: BodyCommand,
        emotion_weights: Dict[str, float],
        emotion_name: str = "NEUTRAL",
        pain_level: float = 0.0,
        dt: float = 0.05,
    ) -> Dict[str, object]:
        """
        Execute a full body command.
        Returns a dict of all joint targets.
        """
        if self._emergency_stop:
            return {"emergency_stop": True}

        result: Dict[str, object] = {}

        if cmd.left_hand:
            result["left_hand"] = self.left_hand.execute(cmd.left_hand, emotion_weights, pain_level)
        if cmd.right_hand:
            result["right_hand"] = self.right_hand.execute(cmd.right_hand, emotion_weights, pain_level)
        if cmd.legs:
            result["legs"] = self.legs.step(cmd.legs, emotion_weights, dt)
        if cmd.head:
            result["head"] = self.head.execute(cmd.head, emotion_name)

        return result

    def status(self) -> str:
        return (f"L-{self.left_hand.describe()} | "
                f"R-{self.right_hand.describe()} | "
                f"{self.legs.describe()}")
