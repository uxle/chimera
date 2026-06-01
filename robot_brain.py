"""
ChimeraRobot Brain — Master Orchestrator
-----------------------------------------
The central nervous system that ties EVERYTHING together:

  Perception Loop  → Sense → Fuse → Understand → Feel → Decide → Act

Every control cycle (~50ms):
  1. Read all sensors (vision, audio, touch, proprioception)
  2. Fuse into unified world embedding
  3. Update emotional state
  4. Compute curiosity signal
  5. Run language model to generate:
       a. Natural language thought / response
       b. Motor action vector
  6. Execute motor commands
  7. Store experience for self-improvement
  8. Learn from prediction errors

This is what makes ChimeraRobot genuinely different from a chatbot:
it is an EMBODIED AGENT — thinking, feeling, and acting in the world.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_config import RobotConfig
from emotion import EmotionSystem, STIMULI
from vision import VisionEncoder, MockCamera, RealCamera, VisualScene
from hearing import HearingSystem, MockMicrophone, RealMicrophone, AudioScene
from touch import TouchSystem, MockTouchSensor, TouchScene
from multimodal import MultimodalFusion, SensoryFrame
from motor import BodyController, BodyCommand, HandCommand, LegCommand, HeadCommand
from motor import HandGesture, GaitMode
from curiosity import CuriosityDrive, AttentionSpotlight


# ─────────────────────────────────────────────────────────────────────
#  Action decoder (brain output → motor commands)
# ─────────────────────────────────────────────────────────────────────

class ActionDecoder(nn.Module):
    """
    Decodes the language model's hidden state into structured motor commands.

    The brain outputs a continuous action vector.
    This module maps that vector to:
      - Hand gesture selection (left + right)
      - Leg gait + speed
      - Head direction
      - Speech intent (should we talk?)
      - Emotional expression
    """

    def __init__(self, d_model: int = 512, n_action_dims: int = 32):
        super().__init__()
        self.n_action_dims = n_action_dims

        # Action head: from brain's last hidden state → continuous action
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.SiLU(),
            nn.Linear(256, n_action_dims),
            nn.Tanh(),   # Outputs in [-1, 1]
        )

        # Discrete action classification heads
        n_gestures = 9    # len(HandGesture)
        n_gaits    = 9    # len(GaitMode)

        self.gesture_head = nn.Linear(n_action_dims, n_gestures)
        self.gait_head    = nn.Linear(n_action_dims, n_gaits)
        self.speech_head  = nn.Linear(n_action_dims, 2)    # speak / silent
        self.force_head   = nn.Sequential(nn.Linear(n_action_dims, 1), nn.Sigmoid())
        self.speed_head   = nn.Sequential(nn.Linear(n_action_dims, 1), nn.Sigmoid())
        self.head_pan     = nn.Sequential(nn.Linear(n_action_dims, 1), nn.Tanh())
        self.head_tilt    = nn.Sequential(nn.Linear(n_action_dims, 1), nn.Tanh())

    def forward(self, brain_state: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:  brain_state: (d_model,) last hidden state from language model
        Returns:
            action_vec: (n_action_dims,) continuous action
            parsed:     dict of discrete commands
        """
        act = self.action_head(brain_state.unsqueeze(0)).squeeze(0)

        act_b = act.unsqueeze(0)
        gesture_logits = self.gesture_head(act_b)
        gait_logits    = self.gait_head(act_b)
        speech_logits  = self.speech_head(act_b)

        gesture_id = int(gesture_logits.argmax(-1))
        gait_id    = int(gait_logits.argmax(-1))
        should_speak = bool(speech_logits.argmax(-1).item())

        parsed = {
            "gesture":     list(HandGesture)[gesture_id],
            "gait":        list(GaitMode)[gait_id],
            "should_speak": should_speak,
            "force":       float(self.force_head(act_b)),
            "speed":       float(self.speed_head(act_b)),
            "head_pan":    float(self.head_pan(act_b)) * 1.5,
            "head_tilt":   float(self.head_tilt(act_b)) * 0.5,
        }
        return act, parsed


# ─────────────────────────────────────────────────────────────────────
#  Experience record
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Experience:
    state_emb:     List[float]         # Fused world embedding at time t
    action_vec:    List[float]         # Action taken
    next_state_emb: List[float]        # World embedding at t+1
    reward:        float               # Extrinsic + intrinsic reward
    emotion_name:  str
    pain_level:    float
    curiosity:     float
    text_thought:  str                 # What the brain "thought"
    timestamp:     float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────
#  Robot brain (main class)
# ─────────────────────────────────────────────────────────────────────

class RobotBrain:
    """
    The complete ChimeraRobot cognitive system.

    Manages:
    - All sensor systems (vision, hearing, touch)
    - Multimodal fusion
    - Language model (ChimeraLLM as the "thinking" module)
    - Emotion system
    - Curiosity drive
    - Motor control
    - Experience logging
    - Self-improvement

    Usage:
        brain = RobotBrain(config)
        brain.start()
        while running:
            output = brain.step(user_input="What do you see?")
            print(output.thought)
            print(f"Moving: {output.body_command}")
    """

    def __init__(self, config: RobotConfig):
        self.config = config
        self.device = self._select_device()
        print(f"[RobotBrain] Initialising on {self.device.upper()}")

        # ── Sensor systems ────────────────────────────────────────────
        self._init_sensors(config)

        # ── Processing modules ────────────────────────────────────────
        self.emotion    = EmotionSystem(config.emotion)
        self.fusion     = MultimodalFusion(config.brain).to(self.device)
        self.curiosity  = CuriosityDrive(
            d_model=config.brain.d_model,
            n_action_dims=config.brain.n_action_dims,
        )
        self.spotlight  = AttentionSpotlight(config.brain.d_model)
        self.decoder    = ActionDecoder(
            config.brain.d_model,
            config.brain.n_action_dims,
        ).to(self.device)
        self.motor      = BodyController(config.left_hand, config.legs)

        # ── Language brain (ChimeraLLM) ──────────────────────────────
        self._llm       = None   # Loaded lazily to save startup time
        self._tokenizer = None

        # ── State ─────────────────────────────────────────────────────
        self._prev_world_emb: Optional[torch.Tensor] = None
        self._prev_action:    Optional[torch.Tensor] = None
        self._last_scene:     Optional[VisualScene]  = None
        self._last_audio:     Optional[AudioScene]   = None
        self._last_touch:     Optional[TouchScene]   = None
        self._step_count      = 0
        self._experiences: List[Experience] = []
        self._running = False

        # ── Experience log ────────────────────────────────────────────
        os.makedirs(os.path.dirname(config.experience_log) or ".", exist_ok=True)

        print(f"[RobotBrain] Ready | simulation={config.simulation_mode}")

    def _select_device(self) -> str:
        d = self.config.device
        if d == "auto":
            if torch.cuda.is_available():   return "cuda"
            if torch.backends.mps.is_available(): return "mps"
            return "cpu"
        return d

    def _init_sensors(self, config: RobotConfig):
        sim = config.simulation_mode
        # Vision
        cam_cls = MockCamera if sim else RealCamera
        self.camera  = cam_cls(config.camera)
        self.vision  = VisionEncoder(config.camera)

        # Hearing
        mic_cls = MockMicrophone if sim else RealMicrophone
        self.microphone = mic_cls(config.audio)
        self.hearing    = HearingSystem(config.audio)

        # Touch
        touch_cls = MockTouchSensor if sim else MockTouchSensor  # Real driver TBD
        self.touch_hw = touch_cls(config.touch)
        self.touch    = TouchSystem(config.touch)

    def load_llm(self, model_path: str, tokenizer_path: str):
        """Load the ChimeraLLM language model. Called after init."""
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ChimeraLLM'))
            from model import ChimeraLM
            from tokenizer import ChimeraTokenizer
            self._llm       = ChimeraLM.load(model_path, device=self.device)
            self._tokenizer = ChimeraTokenizer.load(tokenizer_path)
            print(f"[RobotBrain] LLM loaded: {self._llm.num_parameters()/1e6:.1f}M params")
        except Exception as e:
            print(f"[RobotBrain] Warning: Could not load LLM ({e}). Using mock responses.")

    # ── Sensing ──────────────────────────────────────────────────────

    def _sense_vision(self) -> Optional[VisualScene]:
        frame = self.camera.read()
        if frame is None:
            return None
        return self.vision.encode(frame)

    def _sense_audio(self) -> AudioScene:
        chunk = self.microphone.read_chunk()
        self.hearing.add_samples(chunk)
        return self.hearing.process()

    def _sense_touch(self) -> TouchScene:
        left, right = self.touch_hw.read()
        return self.touch.process(left, right)

    def _sense_proprioception(self) -> torch.Tensor:
        """
        In simulation: return zero proprioception.
        With real hardware: read joint encoders from all 28 joints.
        """
        n = self.config.brain.proprio_feature_dim
        return torch.zeros(n)

    # ── Emotional stimulus from scene ─────────────────────────────────

    def _update_emotion_from_scene(
        self,
        visual:    Optional[VisualScene],
        audio:     AudioScene,
        touch_s:   TouchScene,
    ):
        """Trigger appropriate emotional stimuli based on sensory input."""
        # Touch pain
        if touch_s.pain_level > 0.7:
            self.emotion.receive_stimulus("pain_signal", touch_s.pain_level)
            self.emotion.encode_memory("pain_signal", "negative")
            self.motor.emergency_stop()

        elif touch_s.pain_level > 0.3:
            self.emotion.receive_stimulus("collision", touch_s.pain_level)

        # Visual stimuli
        if visual:
            if visual.motion_magnitude > 0.4:
                self.emotion.receive_stimulus("novel_object", visual.motion_magnitude)

            for det in visual.detections:
                if det.label == "person" and det.confidence > 0.6:
                    self.emotion.receive_stimulus("human_present", det.confidence)
                elif det.label == "face" and det.confidence > 0.7:
                    self.emotion.receive_stimulus("familiar_person", det.confidence)

        # Audio stimuli
        if audio.loudness_db > -20:   # Loud noise
            self.emotion.receive_stimulus("loud_noise",
                                          min(1.0, (audio.loudness_db + 20) / 40))
        if audio.is_speech and audio.speech_prob > 0.7:
            self.emotion.receive_stimulus("human_present", audio.speech_prob * 0.5)

        # Neural update from fused embedding (if available)
        if self._prev_world_emb is not None:
            self.emotion.neural_update(
                self._prev_world_emb.to("cpu")
            )

        self.emotion.step()

    # ── Language generation ───────────────────────────────────────────

    def _generate_thought(
        self,
        world_emb:  torch.Tensor,
        user_input: Optional[str],
        emotion_name: str,
        curiosity_score: float,
        visual_scene:    Optional[VisualScene],
        audio_scene:     AudioScene,
    ) -> str:
        """
        Generate a natural-language thought from the current world state.
        Uses ChimeraLLM if available, otherwise builds a template response.
        """
        if self._llm is not None and self._tokenizer is not None:
            return self._llm_generate(world_emb, user_input, emotion_name)

        # Template fallback (works without a trained LLM)
        return self._template_thought(
            user_input, emotion_name, curiosity_score,
            visual_scene, audio_scene
        )

    def _llm_generate(
        self,
        world_emb:    torch.Tensor,
        user_input:   Optional[str],
        emotion_name: str,
    ) -> str:
        """Use ChimeraLLM to generate a response grounded in sensory context."""
        ctx = self._build_llm_context(user_input, emotion_name)
        ids = self._tokenizer.encode(ctx, add_sos=True, add_eos=False)
        ids_t = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            gen_ids = []
            self._llm.reset_cache()
            out = self._llm(ids_t, use_cache=True)
            logits = out.logits[:, -1, :]

            for _ in range(128):
                probs = F.softmax(logits / 0.7, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                if next_id == self._tokenizer.eos_id:
                    break
                gen_ids.append(next_id)
                next_t = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                out    = self._llm(next_t, use_cache=True)
                logits = out.logits[:, -1, :]
            self._llm.reset_cache()

        return self._tokenizer.decode(gen_ids, skip_special=True)

    def _build_llm_context(self, user_input: Optional[str], emotion: str) -> str:
        parts = [f"[Robot state: feeling {emotion.lower()}]"]
        if self._last_scene and self._last_scene.detections:
            dets = ", ".join(f"{d.label}({d.confidence:.1f})"
                             for d in self._last_scene.detections[:3])
            parts.append(f"[Seeing: {dets}]")
        if self._last_audio and self._last_audio.is_speech:
            parts.append("[Hearing: speech]")
        if self._last_touch and self._last_touch.pain_level > 0.2:
            parts.append(f"[Feeling pain: {self._last_touch.pain_level:.1f}]")
        if user_input:
            parts.append(f"[Human says: {user_input}]")
        parts.append("[Robot responds:]")
        return "\n".join(parts)

    def _template_thought(
        self,
        user_input:      Optional[str],
        emotion_name:    str,
        curiosity_score: float,
        visual:          Optional[VisualScene],
        audio:           AudioScene,
    ) -> str:
        """Rule-based response when no LLM is loaded."""
        parts = []
        em = emotion_name.lower()

        if user_input:
            parts.append(f"I heard you say: '{user_input}'.")

        if visual and visual.detections:
            labels = [d.label for d in visual.detections[:2]]
            parts.append(f"I can see {' and '.join(labels)}.")

        if audio.is_speech:
            parts.append("Someone is speaking to me.")

        emotion_phrases = {
            "joy":     "I'm feeling happy right now!",
            "fear":    "I'm feeling a bit scared.",
            "surprise":"That surprised me!",
            "sadness": "I'm feeling a bit sad.",
            "anger":   "I'm feeling frustrated.",
            "trust":   "I feel comfortable here.",
            "curiosity":"This is interesting, I want to explore!",
        }
        parts.append(emotion_phrases.get(em, f"I'm feeling {em}."))

        if curiosity_score > 0.6:
            parts.append("I notice something new — I want to look closer.")

        return " ".join(parts)

    # ── Main step ─────────────────────────────────────────────────────

    def step(self, user_input: Optional[str] = None) -> "BrainOutput":
        """
        One complete cognitive cycle.

        Returns BrainOutput with:
          - Natural language thought
          - Body command (motor actions)
          - Emotional state
          - Curiosity signal
          - What the robot is attending to
        """
        t_start = time.perf_counter()
        self._step_count += 1

        # ── 1. Sense ─────────────────────────────────────────────────
        visual  = self._sense_vision()
        audio   = self._sense_audio()
        touch_s = self._sense_touch()
        proprio = self._sense_proprioception()

        self._last_scene = visual
        self._last_audio = audio
        self._last_touch = touch_s

        # ── 2. Update emotions ────────────────────────────────────────
        self._update_emotion_from_scene(visual, audio, touch_s)
        emotion_name = self.emotion.dominant_emotion().name
        em_weights   = self.emotion.behavioral_weights()
        em_pad       = self.emotion.state.as_tensor()

        # ── 3. Build sensory frame ────────────────────────────────────
        frame = SensoryFrame(
            vision_emb=visual.embedding if visual else None,
            audio_emb=audio.embedding,
            touch_emb=touch_s.embedding,
            proprio_emb=proprio,
            emotion_pad=em_pad,
            pain_level=touch_s.pain_level,
            is_speech=audio.is_speech,
            motion_mag=visual.motion_magnitude if visual else 0.0,
            n_detections=len(visual.detections) if visual else 0,
        )

        # ── 4. Fuse modalities ────────────────────────────────────────
        world_emb = self.fusion(frame)   # (d_model,)

        # ── 5. Curiosity ──────────────────────────────────────────────
        curiosity_signal = None
        if self._prev_world_emb is not None and self._prev_action is not None:
            curiosity_signal = self.curiosity.compute_curiosity(
                self._prev_world_emb, self._prev_action, world_emb
            )
            # Train curiosity models periodically
            if self._step_count % 10 == 0:
                self.curiosity.learn(self._prev_world_emb, self._prev_action, world_emb)

        curiosity_score = curiosity_signal.intrinsic_reward if curiosity_signal else 0.0
        explore_recommend = (curiosity_signal.recommended_action
                             if curiosity_signal else "exploit")

        # ── 6. Attention spotlight ────────────────────────────────────
        focus_patch = 0
        if visual and visual.patch_tokens is not None and curiosity_signal:
            focus_patch, _ = self.spotlight.select_focus(
                visual.patch_tokens, curiosity_signal, em_weights
            )

        # ── 7. Decode action ──────────────────────────────────────────
        action_vec, action_parsed = self.decoder(world_emb.detach())

        # Emotion overrides to action
        if emotion_name == "FEAR":
            action_parsed["gait"]    = GaitMode.STAND
            action_parsed["gesture"] = HandGesture.OPEN
            action_parsed["speed"]   = 0.1
        elif explore_recommend == "explore":
            action_parsed["gait"]    = GaitMode.WALK_FORWARD
            action_parsed["speed"]  *= 0.7

        # ── 8. Build motor command ────────────────────────────────────
        hand_cmd = HandCommand(
            gesture=action_parsed["gesture"],
            force=action_parsed["force"],
            speed=action_parsed["speed"],
        )
        leg_cmd  = LegCommand(
            gait=action_parsed["gait"],
            speed=action_parsed["speed"] * self.config.legs.max_speed_ms,
        )
        head_cmd = HeadCommand(
            pan=action_parsed["head_pan"],
            tilt=action_parsed["head_tilt"],
        )
        body_cmd = BodyCommand(
            left_hand=hand_cmd,
            right_hand=hand_cmd,
            legs=leg_cmd,
            head=head_cmd,
        )

        # Execute
        joint_targets = self.motor.execute(
            body_cmd, em_weights, emotion_name, touch_s.pain_level
        )

        # ── 9. Generate thought ───────────────────────────────────────
        thought = self._generate_thought(
            world_emb, user_input, emotion_name,
            curiosity_score, visual, audio
        )

        # ── 10. Store experience ──────────────────────────────────────
        if self._prev_world_emb is not None:
            exp = Experience(
                state_emb=self._prev_world_emb.detach().tolist()[:16],  # Truncate for storage
                action_vec=action_vec.detach().tolist(),
                next_state_emb=world_emb.detach().tolist()[:16],
                reward=curiosity_score + (0.5 if action_parsed["should_speak"] else 0.0),
                emotion_name=emotion_name,
                pain_level=touch_s.pain_level,
                curiosity=curiosity_score,
                text_thought=thought,
            )
            self._experiences.append(exp)
            if len(self._experiences) >= 100:
                self._flush_experiences()

        self._prev_world_emb = world_emb.detach()
        self._prev_action    = action_vec.detach()

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return BrainOutput(
            thought=thought,
            body_command=body_cmd,
            joint_targets=joint_targets,
            emotion=emotion_name,
            emotion_summary=str(self.emotion.state),
            curiosity=curiosity_score,
            explore_mode=(explore_recommend == "explore"),
            visual_focus_patch=focus_patch,
            should_speak=action_parsed["should_speak"],
            pain_level=touch_s.pain_level,
            detections=visual.detections if visual else [],
            is_speech_detected=audio.is_speech,
            step_ms=round(elapsed_ms, 1),
            step_count=self._step_count,
        )

    def _flush_experiences(self):
        """Write buffered experiences to disk."""
        with open(self.config.experience_log, "a") as f:
            for exp in self._experiences:
                f.write(json.dumps(exp.to_dict()) + "\n")
        self._experiences.clear()

    def stats(self) -> dict:
        return {
            "steps":     self._step_count,
            "emotion":   str(self.emotion.state),
            "curiosity": self.curiosity.stats(),
            "motor":     self.motor.status(),
        }


# ─────────────────────────────────────────────────────────────────────
#  Brain output
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BrainOutput:
    thought:             str
    body_command:        BodyCommand
    joint_targets:       dict
    emotion:             str
    emotion_summary:     str
    curiosity:           float
    explore_mode:        bool
    visual_focus_patch:  int
    should_speak:        bool
    pain_level:          float
    detections:          list
    is_speech_detected:  bool
    step_ms:             float
    step_count:          int

    def __str__(self) -> str:
        dets = ", ".join(d.label for d in self.detections[:3]) if self.detections else "nothing"
        return (f"[Step {self.step_count} | {self.step_ms:.0f}ms]\n"
                f"  Emotion  : {self.emotion_summary}\n"
                f"  Thought  : {self.thought}\n"
                f"  Seeing   : {dets}\n"
                f"  Speech   : {self.is_speech_detected}\n"
                f"  Curiosity: {self.curiosity:.3f} | explore={self.explore_mode}\n"
                f"  Pain     : {self.pain_level:.2f}\n"
                f"  Motor    : {self.body_command.legs.gait.name if self.body_command.legs else 'still'}")
