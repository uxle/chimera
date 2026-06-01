# ChimeraRobot — Embodied AI System

> A robot that sees, hears, feels, moves, and emotionally evolves —  
> like a baby brain growing up through experience.

---

## The Core Idea

Most AI is **disembodied** — it only processes text.  
ChimeraRobot is **embodied** — it lives in the physical world.

The difference matters because:

| Disembodied AI (LLM) | Embodied AI (ChimeraRobot) |
|---------------------|---------------------------|
| Learns from text    | Learns from experience     |
| No body             | Has hands, legs, sensors   |
| No pain             | Feels pain (protects itself)|
| No curiosity        | Intrinsically curious       |
| No emotions         | Genuine emotional states    |
| No memory of places | Maps the environment        |
| Static personality  | Grows and develops          |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ChimeraRobot Brain                              │
│                                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────┐   │
│  │ Camera  │  │  Micro  │  │  Touch  │  │  Joint Encoders     │   │
│  │ Vision  │  │  Audio  │  │ Sensors │  │  (Proprioception)   │   │
│  └────┬────┘  └────┬────┘  └────┬────┘  └──────────┬──────────┘   │
│       │            │            │                   │              │
│  ┌────▼────┐  ┌────▼────┐  ┌───▼────┐  ┌──────────▼──────────┐   │
│  │VisionEnc│  │AudioEnc │  │TouchEnc│  │  ProprioEncoder     │   │
│  └────┬────┘  └────┬────┘  └───┬────┘  └──────────┬──────────┘   │
│       └────────────┴───────────┴──────────────────┘              │
│                           │                                        │
│                  ┌─────────▼──────────┐                           │
│                  │ Multimodal Fusion  │  ← Cross-modal attention  │
│                  │  (512-dim embed)   │                           │
│                  └─────────┬──────────┘                           │
│                            │                                       │
│         ┌──────────────────┼──────────────────┐                   │
│         │                  │                  │                   │
│   ┌─────▼──────┐   ┌───────▼───────┐  ┌───────▼──────┐          │
│   │  Emotion   │   │  ChimeraLLM   │  │  Curiosity   │          │
│   │  System    │   │  (Reasoning)  │  │  Drive       │          │
│   │ (PAD+MoE)  │   │               │  │ (ICM + Nov.) │          │
│   └─────┬──────┘   └───────┬───────┘  └───────┬──────┘          │
│         └──────────────────┼──────────────────┘                   │
│                            │                                       │
│                   ┌────────▼────────┐                              │
│                   │ Action Decoder  │                              │
│                   └────────┬────────┘                              │
│                            │                                       │
│         ┌──────────────────┼───────────────────┐                  │
│         │                  │                   │                  │
│   ┌─────▼──────┐   ┌───────▼──────┐   ┌───────▼──────┐          │
│   │Hand Control│   │ Leg Control  │   │Head Control  │          │
│   │(10 fingers)│   │ (Biped CPG)  │   │(Pan + Tilt)  │          │
│   └────────────┘   └──────────────┘   └──────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Unique Features

### 🧠 Baby-Like Development (development.py)
The robot starts as a **Newborn** — random movements, basic pain/pleasure.  
As it gains experience, it advances through **Infant → Toddler → Child → Adolescent**.  
Each stage unlocks new capabilities. You cannot skip stages.

| Stage | Steps | Capabilities Unlocked |
|-------|-------|----------------------|
| Newborn | 0 | Touch, basic pain/pleasure |
| Infant | 1K | Vision, audio, curiosity, imitation |
| Toddler | 10K | Walking, grasping, basic language |
| Child | 50K | Planning, empathy, complex emotions |
| Adolescent | 200K | Self-model, theory of mind, emotional regulation |

### 💭 Real Emotions (emotion.py)
PAD (Pleasure-Arousal-Dominance) model with 8 primary emotions.  
Emotions are **not a persona trick** — they change decisions, learning rate, and movement speed.  
- Fear → cautious slow movement, emergency stop if too high  
- Joy → faster, more energetic movement  
- Sadness → slow, shuffling gait, head down  
- Surprise → freeze, reorient, high learning rate  

### 🔍 Genuine Curiosity (curiosity.py)
Intrinsic Curiosity Module (ICM) — the robot is rewarded for visiting new places.  
A Forward Model predicts the next state; prediction error = curiosity reward.  
As the robot explores more, familiar places stop being interesting.  
A Novelty Memory (count-based hashing) tracks which regions are new.

### 🤝 Social Learning (social_learning.py)
- **Imitation**: mirror human actions onto robot body
- **Social referencing**: check human reactions to learn what is safe/dangerous
- **Gaze following**: find what humans are looking at
- **Emotional contagion**: "catch" emotions from nearby humans
- **Facial reward**: infer reward from human smiles/frowns
- **Verbal instruction**: parse spoken commands

### 🗺️ Spatial Memory (embodied_memory.py)
An occupancy grid map built from experience.  
The robot knows where objects are, where it has been, where walls are.  
When a human says "go to the table", the robot knows where it is.

### 🎯 Skill Library (embodied_memory.py)
Procedural memory: learned action sequences that worked.  
"Pick up cup" = built-in skill with success/failure tracking.  
New skills learned from successful episodes are added automatically.

---

## File Structure

```
ChimeraRobot/
├── robot_config.py        Hardware + brain configuration
├── emotion.py             PAD model, Plutchik emotions, emotional memory
├── vision.py              Camera, ViT-style patch encoder, object detector
├── hearing.py             Microphone, MFCC extraction, audio encoder
├── touch.py               Tactile sensors, pain processor, grip quality
├── motor.py               Hand controller, biped CPG, head controller
├── multimodal.py          Cross-modal fusion of all senses (512-dim)
├── curiosity.py           ICM forward model, novelty memory, attention spotlight
├── embodied_memory.py     Sensorimotor buffer, episodic memory, spatial map, skills
├── robot_brain.py         Master orchestrator, main step() loop
├── simulator.py           Full physics-aware robot world simulator
├── social_learning.py     Imitation, gaze following, emotional contagion
├── development.py         Baby-like developmental stages
├── reward.py              Pain/pleasure/social/task/curiosity reward aggregator
├── speech_io.py           Offline TTS (eSpeak/pyttsx3) + STT (Vosk/Whisper)
├── robot_train.py         IL + PPO RL embodied training loop
├── robot_chat.py          Interactive terminal interface
└── requirements_robot.txt
```

---

## Installation

```bash
pip install torch numpy
pip install opencv-python Pillow        # Vision
pip install pyaudio pyttsx3            # Audio / speech
pip install vosk                        # Offline speech recognition
# Linux: sudo apt install espeak-ng portaudio19-dev
```

---

## Quick Start

### 1 — Run in simulation (no hardware needed)

```bash
# Start the interactive terminal (simulation mode)
python robot_chat.py

# With auto-stepping (brain runs continuously)
python robot_chat.py --auto_step

# Type commands:
# /world        → see ASCII world view
# /status       → full sensor + emotion display
# /step 10      → run 10 brain steps
# /emotion      → detailed emotional state
# /memory       → what the robot remembers
# Walk forward please   → natural language command
```

### 2 — Watch the robot develop

```bash
# Run many steps and watch stage advancement
python -c "
from robot_config import RobotConfig
from robot_brain import RobotBrain
from development import DevelopmentalSystem

config = RobotConfig(simulation_mode=True)
brain  = RobotBrain(config)
dev    = DevelopmentalSystem()

for step in range(2000):
    output = brain.step()
    dev.step(step, {'avg_reward': output.curiosity})
    if step % 200 == 0:
        print(dev.report())
"
```

### 3 — Train with RL

```bash
python robot_train.py --mode rl --steps 100000
```

### 4 — Connect to real hardware

```python
from robot_config import RobotConfig

config = RobotConfig(simulation_mode=False)  # Use real hardware
# Robot will use:
#   - OpenCV (cv2) for real camera
#   - PyAudio for real microphone
#   - pyserial for servo controllers
```

---

## Connecting Real Hardware

### Camera
```python
config.camera.device_id = 0        # /dev/video0 on Linux
config.camera.width  = 640
config.camera.height = 480
```

### Microphone
```python
config.audio.device_id = 0         # First mic
config.audio.sample_rate = 16000
```

### Servo Controllers (hands/legs)
Real servo/motor integration requires implementing `ServoInterface` in `motor.py`.  
The current code outputs joint position lists — wire these to your servo driver.  
Common options: Dynamixel SDK, ODrive, Arduino+PCA9685.

---

## Emotional System Details

The PAD (Pleasure-Arousal-Dominance) model maps to 8 primary emotions:

```
         HIGH AROUSAL
              │
    FEAR──────┼──────SURPRISE
   (-P,+A,-D) │      (+P,+A,-D)
              │
DISGUST───────┼───────────TRUST
(-P,0,+D)   NEUTRAL   (+P,-A,+D)
              │
    ANGER─────┼──────JOY
   (-P,+A,+D) │   (+P,+A,+D)
              │
              │
         LOW AROUSAL
```

Emotions influence:
- **Movement**: fear → slow + careful, joy → fast + energetic
- **Learning**: surprise/fear → fast learning (2×), sadness → slow (0.7×)
- **Voice**: fear → higher pitch + faster, sadness → lower + slower
- **Attention**: fear → focus on threat, curiosity → scan for new things

---

## Self-Improvement Loop

Every 20 interactions (by default):
1. Auto-score recent responses by perplexity + repetition
2. Collect user ratings from `/rate +1` / `/rate -1`
3. Fine-tune on highest-rated exchanges (few gradient steps)
4. Adjust temperature based on confidence calibration

The robot also improves through:
- Curiosity-driven exploration (visits new states → builds world model)
- Competence tracking (detects when skills improve/degrade)
- Emotional memory (learns to fear/enjoy things based on outcomes)
- Social learning (imitation, gaze following, facial reward signals)

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores, 2GHz | 8+ cores |
| RAM | 4 GB | 16 GB |
| GPU | Optional | NVIDIA 4GB+ VRAM |
| Storage | 2 GB | 20 GB |
| Camera | 720p 30fps | 1080p 60fps |
| Microphone | Built-in | Directional USB |

Works on:
- Raspberry Pi 4/5 (simulation, micro model)
- Laptop / Desktop CPU (small-medium model)
- PC + NVIDIA GPU (fast training, large model)
- Apple M1/M2/M3 (MPS acceleration)

---

## Limitations

- The emotional system influences decisions but is not "real" emotion
- Speech recognition (Vosk) works best in quiet environments
- Walking balance requires real IMU + proprioception tuning for real hardware
- The LLM's knowledge comes from training data, not from embodied experience
- Object detection without a trained model produces random outputs (needs training data)
- Developmental stages require many steps — real development is much slower

---

## Roadmap

- [ ] Reinforcement learning from human feedback (RLHF) integration
- [ ] Multi-robot social learning (robots teaching each other)
- [ ] Long-term episodic memory with vector search
- [ ] ROS2 integration for real robot middleware
- [ ] Mobile app for remote monitoring and rating
- [ ] Dream/replay during idle (sleep consolidation)
- [ ] Music/rhythm response in emotional system
- [ ] Face recognition for individual human identification
- [ ] Sign language input/output

---

*ChimeraRobot — where embodied intelligence meets emotional depth.*
