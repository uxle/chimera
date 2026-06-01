"""
ChimeraRobot Terminal Interface
---------------------------------
Talk to the robot, issue commands, and watch it think and feel in real time.

Features:
  • Natural language control ("Pick up the cup", "How are you feeling?")
  • Real-time emotional state display
  • Live sensor feed (what the robot sees/hears/touches)
  • Simulator visualisation (ASCII world view)
  • Command mode (direct motor commands without language)
  • Memory inspection (what has the robot remembered?)
  • Self-improvement triggers

Run:
    python robot_chat.py                   # Simulation mode
    python robot_chat.py --real            # Real hardware
    python robot_chat.py --llm checkpoints/best.pt.gz
"""

import argparse
import os
import sys
import threading
import time
from typing import Optional

from robot_config import RobotConfig
from robot_brain import RobotBrain, BrainOutput
from simulator import RobotSimulator
from speech_io import SpeechSystem
from embodied_memory import EmbodiedMemory, SensorySnapshot

try:
    _COLOUR = sys.stdout.isatty()
    def _c(code, t): return f"\033[{code}m{t}\033[0m" if _COLOUR else t
    def green(t):  return _c("32", t)
    def yellow(t): return _c("33", t)
    def cyan(t):   return _c("36", t)
    def red(t):    return _c("31", t)
    def bold(t):   return _c("1",  t)
    def dim(t):    return _c("2",  t)
    def magenta(t):return _c("35", t)
except Exception:
    def green(t): return t
    def yellow(t): return t
    def cyan(t): return t
    def red(t): return t
    def bold(t): return t
    def dim(t): return t
    def magenta(t): return t


BANNER = """
  ╔═══════════════════════════════════════════════════════╗
  ║      ChimeraRobot — Embodied Intelligence System      ║
  ║   Vision · Hearing · Touch · Motion · Emotion · Mind  ║
  ╚═══════════════════════════════════════════════════════╝
"""

HELP = """
  ┌──────────────────────────────────────────────────────────┐
  │  ChimeraRobot Commands                                   │
  ├──────────────────────┬───────────────────────────────────┤
  │ /help                │ Show this help                    │
  │ /status              │ Full sensor + emotion status      │
  │ /emotion             │ Detailed emotional state          │
  │ /memory              │ Show memory stats                 │
  │ /map                 │ Show spatial memory map           │
  │ /skills              │ List learned skills               │
  │ /world               │ Show simulator ASCII view         │
  │ /curiosity           │ Show curiosity stats              │
  │ /step [n]            │ Run n brain steps                 │
  │ /move <dir> [speed]  │ Move: forward/back/left/right     │
  │ /grab                │ Grab nearest object               │
  │ /drop                │ Drop held object                  │
  │ /wave                │ Wave hand                         │
  │ /look <pan> <tilt>   │ Point head (degrees)              │
  │ /rate <+1/-1>        │ Rate last response                │
  │ /episode             │ Close current episode + save      │
  │ /speak <text>        │ Force robot to say something      │
  │ /exit                │ Exit                              │
  └──────────────────────┴───────────────────────────────────┘
"""

EMOTION_ICONS = {
    "JOY":          "😊",
    "TRUST":        "🤝",
    "FEAR":         "😨",
    "SURPRISE":     "😲",
    "SADNESS":      "😢",
    "DISGUST":      "🤢",
    "ANGER":        "😠",
    "ANTICIPATION": "🤔",
    "NEUTRAL":      "😐",
}


class RobotChatInterface:
    """
    Interactive terminal for communicating with and observing ChimeraRobot.
    """

    def __init__(self, args: argparse.Namespace):
        self.args    = args
        self.config  = RobotConfig(simulation_mode=not args.real)
        self.brain   = RobotBrain(self.config)
        self.memory  = EmbodiedMemory(self.config.memory_db)
        self.speech  = SpeechSystem(self.config.speech, simulation=not args.real)
        self.sim     = RobotSimulator(self.config) if self.config.simulation_mode else None

        # Load LLM if provided
        if args.llm and os.path.exists(args.llm):
            tok = args.tokenizer or args.llm.replace(".pt.gz", "_tokenizer.json")
            if os.path.exists(tok):
                self.brain.load_llm(args.llm, tok)

        self._last_output: Optional[BrainOutput] = None
        self._step_count  = 0
        self._running     = False
        self._auto_step   = args.auto_step

        # Background step thread (runs brain continuously)
        self._bg_thread: Optional[threading.Thread] = None

    # ── Background brain loop ─────────────────────────────────────────

    def _bg_loop(self):
        """Runs brain steps in background (one step every control_loop period)."""
        period = 1.0 / self.config.control_loop_hz
        while self._running:
            t0 = time.perf_counter()
            try:
                output = self.brain.step()
                self._last_output = output
                self._step_count  += 1

                # Record to memory
                snap = SensorySnapshot(
                    step=self._step_count,
                    timestamp=time.time(),
                    world_emb=(self.brain._prev_world_emb.tolist()[:8]
                               if self.brain._prev_world_emb is not None else []),
                    emotion=output.emotion,
                    pain=output.pain_level,
                    detections=output.detections[:3] if output.detections else [],
                    is_speech=output.is_speech_detected,
                    action_taken=[],
                )
                self.memory.record_step(snap)

                # Auto-speak if robot decides to and has something to say
                if output.should_speak and output.thought:
                    self.speech.speak(output.thought, output.emotion, blocking=False)

            except Exception as e:
                if self._running:
                    print(f"\n[BG Error] {e}")
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, period - elapsed)
            time.sleep(sleep_t)

    def start_bg(self):
        self._running  = True
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def stop_bg(self):
        self._running = False
        if self._bg_thread:
            self._bg_thread.join(timeout=2.0)

    # ── Manual step ──────────────────────────────────────────────────

    def step(self, user_input: Optional[str] = None, n: int = 1) -> BrainOutput:
        """Run n brain steps with optional user input."""
        output = None
        for i in range(n):
            inp    = user_input if i == 0 else None
            output = self.brain.step(inp)
            self._last_output = output
            self._step_count  += 1
        return output

    # ── Command handlers ─────────────────────────────────────────────

    def _cmd_status(self):
        if not self._last_output:
            print(yellow("  No output yet — run /step first"))
            return
        o = self._last_output
        icon = EMOTION_ICONS.get(o.emotion, "🤖")
        print(bold("\n  ── Sensor Status ─────────────────────────────────────────"))
        print(f"  {icon} Emotion  : {cyan(o.emotion_summary)}")
        print(f"  👁 Seeing   : {', '.join(d.label for d in o.detections) or 'nothing'}")
        print(f"  👂 Hearing  : {'speech detected' if o.is_speech_detected else 'quiet'}")
        print(f"  ✋ Pain     : {'⚠ ' + red(f'{o.pain_level:.2f}') if o.pain_level > 0.2 else green('0.00')}")
        print(f"  🔍 Curiosity: {o.curiosity:.3f} {'(exploring)' if o.explore_mode else '(exploiting)'}")
        print(f"  💬 Thought  : {o.thought}")
        print(f"  ⚙  Step     : {o.step_count} | {o.step_ms:.0f}ms/step")
        print(bold("  ─────────────────────────────────────────────────────────\n"))

    def _cmd_emotion(self):
        print(bold("\n  ── Emotional State ───────────────────────────────────────"))
        print(f"  {self.brain.emotion.summary()}")
        print(bold("  ─────────────────────────────────────────────────────────\n"))

    def _cmd_memory(self):
        stats = self.memory.stats()
        ep    = self.memory.episodic.recent(n=3)
        print(bold("\n  ── Memory ────────────────────────────────────────────────"))
        print(f"  Sensorimotor frames : {stats['sensorimotor_frames']}")
        print(f"  Total episodes      : {stats['total_episodes']}")
        print(f"  Success rate        : {stats['success_rate']*100:.1f}%")
        print(f"  Map coverage        : {stats['map_coverage_%']}%")
        print(f"  Known skills        : {stats['total_skills']}")
        if ep:
            print(f"\n  Recent episodes:")
            for e in ep:
                flag = green("✓") if e.success else red("✗")
                print(f"    {flag} [{e.episode_id}] {e.summary[:60]}  "
                      f"[{e.dominant_emotion}, pain={e.avg_pain:.2f}]")
        print(bold("  ─────────────────────────────────────────────────────────\n"))

    def _cmd_map(self):
        sm = self.memory.spatial
        local = sm.local_map(radius_cells=15)
        print(bold("\n  ── Spatial Map (local) ──────────────────────────────────"))
        rows = []
        for row in local:
            cells = []
            for cell in row:
                if cell < 0.1:     cells.append(green("."))
                elif cell > 0.9:   cells.append(red("█"))
                elif cell == 0.3:  cells.append(yellow("★"))
                else:              cells.append(dim("?"))
            rows.append("".join(cells))
        print("\n".join("  " + r for r in rows))
        objs = list(sm._objects.values())
        if objs:
            print(f"\n  Known objects: {', '.join(set(objs))}")
        print(f"  Coverage: {sm.exploration_coverage()*100:.1f}%")
        print(bold("  ─────────────────────────────────────────────────────────\n"))

    def _cmd_skills(self):
        skills = self.memory.skills._conn.execute(
            "SELECT name, description, success_count, failure_count FROM skills"
        ).fetchall()
        print(bold("\n  ── Skill Library ─────────────────────────────────────────"))
        for name, desc, succ, fail in skills:
            rate = succ / max(1, succ + fail)
            bar  = green("█") * int(rate * 10) + dim("░") * (10 - int(rate * 10))
            print(f"  {bar} {bold(name):<25} {desc[:40]}")
        print(bold("  ─────────────────────────────────────────────────────────\n"))

    def _cmd_world(self):
        if self.sim:
            print(self.sim.render_ascii())
        else:
            print(yellow("  World view only available in simulation mode"))

    def _cmd_curiosity(self):
        stats = self.brain.curiosity.stats()
        print(bold("\n  ── Curiosity System ──────────────────────────────────────"))
        print(f"  Average curiosity  : {stats['avg_curiosity']:.4f}")
        print(f"  Should explore     : {green('yes') if stats['should_explore'] else 'no'}")
        print(f"  Map coverage       : {stats['coverage_%']}%")
        print(f"  Buckets visited    : {stats['buckets_visited']}/{stats['total_buckets']}")
        print(f"  Total visits       : {stats['total_visits']}")
        print(bold("  ─────────────────────────────────────────────────────────\n"))

    def _cmd_move(self, args: str):
        parts = args.strip().split()
        if not parts:
            print(red("  Usage: /move <forward|back|left|right> [speed]"))
            return
        direction = parts[0].lower()
        speed = float(parts[1]) if len(parts) > 1 else 0.5
        from motor import GaitMode, LegCommand
        gait_map = {
            "forward": GaitMode.WALK_FORWARD,
            "back":    GaitMode.WALK_BACK,
            "left":    GaitMode.TURN_LEFT,
            "right":   GaitMode.TURN_RIGHT,
        }
        gait = gait_map.get(direction, GaitMode.STAND)
        self.brain.motor.legs.step(
            LegCommand(gait=gait, speed=speed),
            self.brain.emotion.behavioral_weights(),
            dt=0.5
        )
        print(green(f"  Moving {direction} at speed {speed:.1f}"))

    def _cmd_episode(self):
        summary = (f"Episode at step {self._step_count}. "
                   f"Dominant emotion: {self.memory.sensorimotor.dominant_recent_emotion()}. "
                   f"Seen: {', '.join(set(self._last_output.detections[:3]) if self._last_output and self._last_output.detections else ['nothing'])}.")
        self.memory.close_episode(summary)
        print(green(f"  ✓ Episode saved: {summary[:80]}"))

    def _cmd_rate(self, val: str):
        try:
            r = float(val.strip())
        except ValueError:
            r = 1.0 if "+" in val else -1.0
        self.brain.emotion.receive_stimulus(
            "human_praise" if r > 0 else "human_scolding", abs(r)
        )
        self.brain.emotion.encode_memory(
            "human_praise" if r > 0 else "human_scolding",
            "positive" if r > 0 else "negative"
        )
        icon = green("👍") if r > 0 else red("👎")
        print(f"  {icon} Feedback recorded (robot emotional update applied)")

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self):
        print(cyan(BANNER))
        print(bold("  Initialising robot brain..."))

        if self._auto_step:
            self.start_bg()
            print(green("  ✓ Background brain loop started\n"))
        else:
            print(dim("  Manual mode: use /step [n] to advance the brain\n"))

        print(dim("  Type /help for commands | Type naturally to talk to the robot\n"))

        while True:
            try:
                prompt = f"  {bold('You')}: "
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                self._on_exit()
                break

            if not user_input:
                continue

            # Command dispatch
            if user_input.startswith("/"):
                parts = user_input[1:].split(maxsplit=1)
                cmd   = parts[0].lower()
                arg   = parts[1] if len(parts) > 1 else ""

                if cmd in ("exit", "quit", "q"):
                    self._on_exit()
                    break
                elif cmd == "help":           print(HELP)
                elif cmd == "status":         self._cmd_status()
                elif cmd == "emotion":        self._cmd_emotion()
                elif cmd == "memory":         self._cmd_memory()
                elif cmd == "map":            self._cmd_map()
                elif cmd == "skills":         self._cmd_skills()
                elif cmd == "world":          self._cmd_world()
                elif cmd == "curiosity":      self._cmd_curiosity()
                elif cmd == "episode":        self._cmd_episode()
                elif cmd == "rate":           self._cmd_rate(arg)
                elif cmd == "move":           self._cmd_move(arg)
                elif cmd == "step":
                    n = int(arg) if arg.isdigit() else 1
                    o = self.step(n=n)
                    print(f"\n{o}\n")
                elif cmd == "grab":
                    o = self.step("Grab the nearest object")
                    print(f"  {green('Chimera')}: {o.thought}")
                elif cmd == "drop":
                    o = self.step("Release what I am holding")
                    print(f"  {green('Chimera')}: {o.thought}")
                elif cmd == "wave":
                    o = self.step("Wave hello to the person")
                    print(f"  {green('Chimera')}: {o.thought}")
                elif cmd == "look":
                    vals = arg.split()
                    pan  = float(vals[0]) if vals else 0.0
                    tilt = float(vals[1]) if len(vals) > 1 else 0.0
                    import math
                    self.brain.motor.head.execute(
                        __import__("motor").HeadCommand(
                            pan=math.radians(pan), tilt=math.radians(tilt)
                        ),
                        self.brain.emotion.dominant_emotion().name,
                    )
                    print(green(f"  Head → pan={pan}° tilt={tilt}°"))
                elif cmd == "speak":
                    txt = arg.strip()
                    self.speech.speak(txt, self.brain.emotion.dominant_emotion().name)
                else:
                    print(red(f"  Unknown command: /{cmd}  (type /help)"))
                continue

            # Natural language input → brain step
            if self._auto_step:
                # Inject into the next background step
                output = self.step(user_input=user_input)
            else:
                output = self.step(user_input=user_input)

            icon = EMOTION_ICONS.get(output.emotion, "🤖")
            print(f"\n  {icon} {cyan(bold('Chimera'))}: {output.thought}\n")

            if output.pain_level > 0.5:
                print(f"  {red('⚠ WARNING: High pain level!')}")

            if output.explore_mode:
                print(f"  {dim('[Curiosity: exploring new territory]')}")

            # Speak response
            self.speech.speak(output.thought, output.emotion)

    def _on_exit(self):
        print(cyan("\n  Shutting down robot brain..."))
        self.stop_bg()
        self._cmd_episode()
        self.memory.episodic._conn.close()
        self.memory.skills._conn.close()
        self.speech.stop_speaking()
        print(dim(f"  Session: {self._step_count} steps completed"))
        print(cyan("  Goodbye!\n"))


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ChimeraRobot interactive terminal")
    p.add_argument("--real",       action="store_true", help="Use real hardware")
    p.add_argument("--llm",        type=str, default=None, help="Path to trained LLM")
    p.add_argument("--tokenizer",  type=str, default=None)
    p.add_argument("--auto_step",  action="store_true",
                   help="Run brain continuously in background")
    p.add_argument("--hz",         type=int, default=5,
                   help="Background step frequency (Hz, default=5)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    RobotChatInterface(args).run()
