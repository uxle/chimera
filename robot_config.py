"""
ChimeraRobot Configuration
---------------------------
All hardware specs, sensor parameters, and brain settings.
Designed to work with real hardware OR a software simulator.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
#  Sensor configurations
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    width: int  = 224          # Image width (pixels)
    height: int = 224          # Image height (pixels)
    fps: int    = 30           # Frames per second
    channels: int = 3          # RGB
    fov_degrees: float = 90.0  # Field of view
    device_id: int = 0         # Camera index (0 = first camera)
    # Visual encoder
    feature_dim: int = 512     # Output embedding size
    patch_size: int  = 16      # ViT-style patch size


@dataclass
class AudioConfig:
    sample_rate: int    = 16000  # Hz
    chunk_size: int     = 1024   # Samples per buffer read
    n_mfcc: int         = 40     # MFCC feature count
    n_fft: int          = 512    # FFT window size
    hop_length: int     = 256    # FFT hop
    channels: int       = 1      # Mono
    device_id: int      = 0
    # Audio encoder output
    feature_dim: int    = 256
    buffer_seconds: float = 2.0  # How much audio to process at once


@dataclass
class TouchConfig:
    """
    Tactile sensors on hands.
    Each finger tip has a pressure sensor + temperature sensor.
    """
    n_fingers: int       = 5       # Per hand
    n_hands: int         = 2
    n_sensors_per_finger: int = 4  # Pressure at different points
    max_force_n: float   = 50.0    # Newtons
    temperature_range: Tuple[float, float] = (0.0, 100.0)  # Celsius
    feature_dim: int     = 64      # Touch encoder output


@dataclass
class ProprioceptionConfig:
    """
    Internal body sense: joint angles, velocities, accelerations, balance.
    """
    # Arm joints
    n_arm_joints: int   = 6    # Per arm (shoulder x3, elbow, wrist x2)
    n_arms: int         = 2
    # Leg joints
    n_leg_joints: int   = 6    # Per leg (hip x3, knee, ankle x2)
    n_legs: int         = 2
    # Head
    n_head_joints: int  = 2    # Pan, tilt
    # IMU (Inertial Measurement Unit)
    has_imu: bool       = True   # Accelerometer + gyroscope
    # Feature output
    feature_dim: int    = 128


# ─────────────────────────────────────────────────────────────────────
#  Actuator configurations
# ─────────────────────────────────────────────────────────────────────

@dataclass
class HandConfig:
    n_fingers: int       = 5
    n_joints_per_finger: int = 3   # Base, middle, tip
    n_wrist_joints: int  = 2       # Flex, rotate
    max_torque_nm: float = 5.0     # Newton-metres
    grip_force_n: float  = 30.0    # Max grip force
    # Control
    control_freq_hz: int = 50      # Control loop frequency
    position_resolution: int = 4096  # 12-bit servo


@dataclass
class LegConfig:
    n_legs: int         = 2        # Biped
    n_joints_per_leg: int = 6
    max_torque_nm: float = 80.0    # Legs need more torque
    max_speed_ms: float = 1.5      # m/s walking speed
    step_height_m: float = 0.08    # How high to lift foot
    control_freq_hz: int = 100     # Higher for balance


@dataclass
class SpeechConfig:
    tts_engine: str  = "espeak"    # "espeak" (offline) or "pyttsx3"
    stt_engine: str  = "vosk"      # "vosk" (offline) or "whisper"
    language: str    = "en"
    voice_speed: int = 150         # Words per minute
    voice_pitch: int = 50          # 0-100


# ─────────────────────────────────────────────────────────────────────
#  Brain (LLM) configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BrainConfig:
    """
    The language/reasoning brain that integrates all modalities.
    Inherits from ChimeraLLM architecture but extended for multimodal input.
    """
    # Language model
    vocab_size: int       = 16000
    d_model: int          = 512        # Must match all encoder output dims after projection
    n_heads: int          = 8
    n_kv_heads: int       = 2
    n_layers: int         = 16
    d_ff: int             = 1792
    context_length: int   = 1024
    # MoE
    use_moe: bool         = True
    moe_layer_freq: int   = 4
    n_experts: int        = 4
    moe_top_k: int        = 2
    moe_aux_loss_weight: float = 0.01
    rms_norm_eps: float   = 1e-6
    rope_base: int        = 10000
    attn_dropout: float   = 0.0
    ffn_dropout: float    = 0.0
    embed_dropout: float  = 0.1
    init_std: float       = 0.02
    quant_bits: Optional[int] = None

    # Multimodal fusion
    vision_feature_dim: int  = 512    # From vision encoder
    audio_feature_dim: int   = 256    # From audio encoder
    touch_feature_dim: int   = 64     # From touch encoder
    proprio_feature_dim: int = 128    # From proprioception encoder

    # Action space
    n_action_dims: int    = 32        # Continuous action output size
    action_vocab_size: int = 256      # Discretised action tokens


# ─────────────────────────────────────────────────────────────────────
#  Emotion configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class EmotionConfig:
    """
    PAD (Pleasure-Arousal-Dominance) emotional model.
    Emotions are continuous 3D vectors, not discrete categories.
    8 primary emotions map onto this space (Plutchik's wheel).
    """
    # Decay rates: how quickly each dimension returns to baseline
    pleasure_decay: float  = 0.95   # Per timestep
    arousal_decay: float   = 0.90   # Arousal fades faster
    dominance_decay: float = 0.97

    # Baseline (resting state)
    baseline_pleasure:  float = 0.1   # Slightly positive
    baseline_arousal:   float = 0.2   # Calm but attentive
    baseline_dominance: float = 0.5   # Neutral control

    # Learning rate for emotion updates
    emotion_lr: float = 0.1

    # Thresholds that trigger behavior changes
    fear_threshold: float   = -0.5   # Pleasure below this → fear response
    joy_threshold: float    = 0.7    # Pleasure above this → joy expression
    anger_threshold: float  = -0.3   # Low dominance + low pleasure → frustration
    curiosity_threshold: float = 0.6 # High arousal + moderate pleasure → explore


# ─────────────────────────────────────────────────────────────────────
#  Full robot configuration
# ─────────────────────────────────────────────────────────────────────

@dataclass
class RobotConfig:
    name: str = "Chimera-Robot-v1"

    # Hardware
    camera:      CameraConfig      = field(default_factory=CameraConfig)
    audio:       AudioConfig       = field(default_factory=AudioConfig)
    touch:       TouchConfig       = field(default_factory=TouchConfig)
    proprio:     ProprioceptionConfig = field(default_factory=ProprioceptionConfig)
    left_hand:   HandConfig        = field(default_factory=HandConfig)
    right_hand:  HandConfig        = field(default_factory=HandConfig)
    legs:        LegConfig         = field(default_factory=LegConfig)
    speech:      SpeechConfig      = field(default_factory=SpeechConfig)

    # Brain
    brain:       BrainConfig       = field(default_factory=BrainConfig)
    emotion:     EmotionConfig     = field(default_factory=EmotionConfig)

    # Runtime
    simulation_mode: bool = True      # If True: use MockHardware instead of real drivers
    control_loop_hz: int  = 20        # Main brain loop frequency
    device: str           = "auto"    # "cpu" | "cuda" | "mps" | "auto"

    # Storage
    checkpoint_dir: str   = "robot_checkpoints"
    memory_db: str        = "robot_memory.db"
    experience_log: str   = "robot_experience.jsonl"

    @classmethod
    def from_json(cls, path: str) -> "RobotConfig":
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        # Simple shallow merge (extend for nested if needed)
        for k, v in data.items():
            if hasattr(cfg, k) and not isinstance(v, dict):
                setattr(cfg, k, v)
        return cfg

    def save_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
