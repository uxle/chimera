"""
ChimeraRobot Simulator
-----------------------
A lightweight robot world simulator for training and testing
without real hardware. Runs entirely in Python.

Simulates:
  • 3D physics (simplified rigid body, gravity, collision)
  • Robot body (joints, mass, moments of inertia)
  • Object interaction (pick up, put down, push)
  • Multiple entities (humans, furniture, obstacles)
  • Sensor simulation (camera, audio, touch, IMU)
  • Reward signals (task success, collision, energy use)

Two rendering modes:
  • Text/ASCII  — always available, fast
  • Matplotlib  — 2D top-down view (pip install matplotlib)
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from robot_config import RobotConfig


# ─────────────────────────────────────────────────────────────────────
#  World entities
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, o): return Vec3(self.x+o.x, self.y+o.y, self.z+o.z)
    def __sub__(self, o): return Vec3(self.x-o.x, self.y-o.y, self.z-o.z)
    def __mul__(self, s): return Vec3(self.x*s,   self.y*s,   self.z*s)
    def dist(self, o)   : return math.sqrt((self.x-o.x)**2+(self.y-o.y)**2+(self.z-o.z)**2)
    def norm(self)      : m=math.sqrt(self.x**2+self.y**2+self.z**2)+1e-9; return Vec3(self.x/m,self.y/m,self.z/m)
    def __repr__(self)  : return f"({self.x:.2f},{self.y:.2f},{self.z:.2f})"


@dataclass
class WorldObject:
    name:      str
    label:     str            # Visual class label
    pos:       Vec3
    size:      Vec3           # Bounding box half-extents
    mass_kg:   float = 1.0
    movable:   bool  = True
    held_by:   Optional[str] = None   # "left" | "right" | None
    color:     str   = "grey"
    velocity:  Vec3  = field(default_factory=Vec3)


@dataclass
class SimHuman:
    name:       str
    pos:        Vec3
    heading:    float = 0.0   # Radians
    speed:      float = 0.5   # m/s
    is_speaking: bool = False
    speech_text: str = ""
    emotion:    str  = "happy"
    waypoints:  List[Vec3] = field(default_factory=list)
    _wp_idx:    int  = 0


@dataclass
class RobotState:
    """Complete state of the simulated robot."""
    pos:          Vec3  = field(default_factory=Vec3)
    heading:      float = 0.0       # Radians
    velocity:     Vec3  = field(default_factory=Vec3)
    # Joint positions (simplified)
    left_arm_pos: Vec3  = field(default_factory=lambda: Vec3(0, 0.4, 1.0))
    right_arm_pos: Vec3 = field(default_factory=lambda: Vec3(0, -0.4, 1.0))
    left_grip:    float = 0.0     # 0=open, 1=closed
    right_grip:   float = 0.0
    head_pan:     float = 0.0     # Radians
    head_tilt:    float = 0.0
    # Held objects
    left_held:    Optional[str] = None
    right_held:   Optional[str] = None
    # Battery
    battery:      float = 1.0     # 0–1
    # Metrics
    total_distance: float = 0.0
    collision_count: int  = 0


# ─────────────────────────────────────────────────────────────────────
#  Physics engine (simplified)
# ─────────────────────────────────────────────────────────────────────

class SimplePhysics:
    """
    Lightweight physics: collision detection, gravity, friction.
    Not a full rigid-body simulator — just good enough for navigation
    and pick-and-place tasks.
    """

    GRAVITY   = 9.81   # m/s²
    FRICTION  = 0.7    # Ground friction coefficient
    RESTITUTION = 0.2  # Bounciness

    def __init__(self, world_bounds: Tuple[float,float,float,float] = (-5,-5,5,5)):
        self.x_min, self.y_min, self.x_max, self.y_max = world_bounds

    def step_object(self, obj: WorldObject, dt: float):
        """Integrate velocity, apply gravity, clamp to floor."""
        if obj.held_by:
            return   # Held objects move with the hand
        obj.velocity.z -= self.GRAVITY * dt
        obj.pos = obj.pos + obj.velocity * dt
        # Floor collision
        if obj.pos.z <= 0:
            obj.pos.z = 0.0
            obj.velocity.z *= -self.RESTITUTION
            obj.velocity.x *= self.FRICTION
            obj.velocity.y *= self.FRICTION
        # World boundary clamp
        obj.pos.x = max(self.x_min, min(self.x_max, obj.pos.x))
        obj.pos.y = max(self.y_min, min(self.y_max, obj.pos.y))

    def check_collision(self, robot: RobotState, objects: List[WorldObject]) -> List[str]:
        """Returns names of objects the robot is colliding with."""
        collisions = []
        r_pos  = robot.pos
        r_size = 0.3   # Robot footprint radius (m)
        for obj in objects:
            if obj.held_by:
                continue
            dist = math.sqrt((r_pos.x - obj.pos.x)**2 + (r_pos.y - obj.pos.y)**2)
            obj_r = max(obj.size.x, obj.size.y)
            if dist < r_size + obj_r:
                collisions.append(obj.name)
        return collisions

    def within_reach(self, robot: RobotState, obj: WorldObject, reach_m: float = 0.7) -> bool:
        arm_pos = robot.right_arm_pos
        return arm_pos.dist(obj.pos) < reach_m


# ─────────────────────────────────────────────────────────────────────
#  Sensor simulators
# ─────────────────────────────────────────────────────────----------------------------------------

class SimCamera:
    """Generates synthetic 'sensor data' from world state."""

    def __init__(self, fov_deg: float = 90.0, max_range: float = 5.0):
        self.fov_rad   = math.radians(fov_deg)
        self.max_range = max_range

    def visible_objects(
        self,
        robot:   RobotState,
        objects: List[WorldObject],
        humans:  List[SimHuman],
    ) -> List[Tuple[str, float, float]]:
        """
        Returns list of (label, distance, angle_rad) for visible entities.
        Entity must be within FOV and max range.
        """
        visible = []
        for obj in objects:
            dx = obj.pos.x - robot.pos.x
            dy = obj.pos.y - robot.pos.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > self.max_range:
                continue
            angle = math.atan2(dy, dx) - robot.heading
            # Normalise angle to [-π, π]
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if abs(angle) < self.fov_rad / 2:
                visible.append((obj.label, dist, angle))

        for human in humans:
            dx = human.pos.x - robot.pos.x
            dy = human.pos.y - robot.pos.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > self.max_range:
                continue
            angle = math.atan2(dy, dx) - robot.heading
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if abs(angle) < self.fov_rad / 2:
                visible.append(("person", dist, angle))

        return visible

    def render_frame(
        self,
        robot:   RobotState,
        objects: List[WorldObject],
        humans:  List[SimHuman],
        img_h:   int = 224,
        img_w:   int = 224,
    ) -> np.ndarray:
        """Render a synthetic camera frame as uint8 numpy array."""
        frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        # Sky
        frame[:img_h//2, :] = [135, 206, 235]
        # Floor
        frame[img_h//2:, :] = [101, 67, 33]

        # Draw visible objects as coloured rectangles
        color_map = {
            "cup":     [200, 100, 50],
            "person":  [255, 200, 150],
            "chair":   [100, 100, 200],
            "table":   [150, 100, 50],
            "obstacle":[200, 50,  50],
            "ball":    [255, 50,  50],
        }
        vis = self.visible_objects(robot, objects, humans)
        for label, dist, angle in vis:
            # Map (angle, dist) → (x, y) in image
            x_norm = 0.5 + angle / self.fov_rad
            y_norm = 0.5 + (dist / self.max_range - 0.5) * 0.5
            px = int(x_norm * img_w)
            py = int(y_norm * img_h)
            size = max(5, int(40 / (dist + 0.5)))
            color = color_map.get(label, [128, 128, 128])
            x0, x1 = max(0, px-size), min(img_w, px+size)
            y0, y1 = max(0, py-size), min(img_h, py+size)
            frame[y0:y1, x0:x1] = color

        # Add noise
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame


# ─────────────────────────────────────────────────────────────────────
#  World simulator
# ─────────────────────────────────────────────────────────────────────

class RobotSimulator:
    """
    Complete robot world simulator.

    Manages:
    - Robot state (position, joints, battery)
    - World objects (cups, chairs, obstacles)
    - Simulated humans (walk around, speak)
    - Physics (collision, gravity)
    - Sensor simulation (camera, audio, touch, IMU)
    - Reward computation (task success, safety violations)

    Usage:
        sim = RobotSimulator(config)
        sim.reset()
        while True:
            obs   = sim.observe()
            action = brain.step(obs)
            reward, done = sim.apply_action(action)
    """

    def __init__(self, config: RobotConfig):
        self.config  = config
        self.physics = SimplePhysics(world_bounds=(-5, -5, 5, 5))
        self.camera  = SimCamera(fov_deg=config.camera.fov_degrees)
        self.dt      = 1.0 / config.control_loop_hz
        self.robot   = RobotState()
        self.objects: List[WorldObject] = []
        self.humans:  List[SimHuman]    = []
        self._step   = 0
        self._tasks: List[dict]         = []
        self._active_task: Optional[dict] = None
        self.reset()

    def reset(self):
        """Reset world to initial state."""
        self._step = 0
        self.robot = RobotState(pos=Vec3(0, 0, 0))

        # Spawn some objects
        self.objects = [
            WorldObject("cup1",      "cup",      Vec3(1.5, 0.2, 0.0),  Vec3(0.05,0.05,0.1),  mass_kg=0.3, color="red"),
            WorldObject("cup2",      "cup",      Vec3(1.5, -0.3, 0.0), Vec3(0.05,0.05,0.1),  mass_kg=0.3, color="blue"),
            WorldObject("chair1",    "chair",    Vec3(2.5, 1.0, 0.0),  Vec3(0.3,0.3,0.5),    mass_kg=5.0, movable=False),
            WorldObject("table1",    "table",    Vec3(3.0, 0.0, 0.0),  Vec3(0.6,0.4,0.4),    mass_kg=20.0, movable=False),
            WorldObject("obstacle1", "obstacle", Vec3(-1.5, 0.5, 0.0), Vec3(0.3,0.3,1.0),    mass_kg=50.0, movable=False),
            WorldObject("ball1",     "ball",     Vec3(0.8, -1.0, 0.0), Vec3(0.1,0.1,0.1),    mass_kg=0.5),
        ]

        # Spawn a human
        self.humans = [
            SimHuman(
                name="person_alice",
                pos=Vec3(2.0, -1.5, 0.0),
                heading=math.pi / 2,
                waypoints=[Vec3(2,2,0), Vec3(-1,2,0), Vec3(-1,-2,0), Vec3(2,-2,0)],
            )
        ]

        # Set up tasks
        self._tasks = [
            {"name": "pick_up_cup1",   "goal": "Pick up the red cup",  "target": "cup1"},
            {"name": "wave_to_person", "goal": "Wave to the person",   "target": "person_alice"},
        ]
        self._active_task = self._tasks[0]
        return self.observe()

    # ── Sensing ──────────────────────────────────────────────────────

    def observe(self) -> dict:
        """Return the full observation dict from all simulated sensors."""
        vis_objects = self.camera.visible_objects(self.robot, self.objects, self.humans)
        frame       = self.camera.render_frame(self.robot, self.objects, self.humans)
        detections  = [(label, dist) for label, dist, _ in vis_objects]

        # Simulated touch: check if hands are near objects
        touch_force_left  = self._hand_contact_force("left")
        touch_force_right = self._hand_contact_force("right")

        # Simulated audio: speech if human is close
        hearing_speech = False
        speech_text    = ""
        for human in self.humans:
            dist = self.robot.pos.dist(human.pos)
            if dist < 2.0 and human.is_speaking:
                hearing_speech = True
                speech_text    = human.speech_text

        # IMU (simplified)
        imu_accel = np.array([0.0, 0.0, 9.81], dtype=np.float32)

        # Battery drain
        self.robot.battery = max(0.0, self.robot.battery - 0.0001)

        return {
            "camera_frame":     frame,
            "detections":       detections,
            "touch_left":       touch_force_left,
            "touch_right":      touch_force_right,
            "hearing_speech":   hearing_speech,
            "speech_text":      speech_text,
            "imu_accel":        imu_accel,
            "robot_pos":        (self.robot.pos.x, self.robot.pos.y),
            "robot_heading":    self.robot.heading,
            "battery":          self.robot.battery,
            "task":             self._active_task["goal"] if self._active_task else "explore",
            "step":             self._step,
        }

    def _hand_contact_force(self, side: str) -> float:
        """Simulate touch sensor: how much force is on the hand?"""
        arm_pos = self.robot.left_arm_pos if side == "left" else self.robot.right_arm_pos
        held    = self.robot.left_held    if side == "left" else self.robot.right_held
        if held:
            return 0.5   # Holding something
        for obj in self.objects:
            if arm_pos.dist(obj.pos) < 0.2:
                return 0.3   # Touching but not gripping
        return 0.0

    # ── Action application ────────────────────────────────────────────

    def apply_action(self, action: dict) -> Tuple[float, bool, dict]:
        """
        Apply a BodyCommand dict to the simulator.
        Returns (reward, done, info).
        """
        self._step += 1
        reward = -0.001   # Small time penalty (encourages efficiency)
        done   = False
        info   = {}

        # Move robot
        gait = str(action.get("gait", "STAND"))
        speed = float(action.get("speed", 0.0)) * self.dt

        if "WALK_FORWARD" in gait:
            dx = speed * math.cos(self.robot.heading)
            dy = speed * math.sin(self.robot.heading)
            self.robot.total_distance += speed
        elif "WALK_BACK" in gait:
            dx = -speed * math.cos(self.robot.heading) * 0.6
            dy = -speed * math.sin(self.robot.heading) * 0.6
        elif "TURN_LEFT" in gait:
            dx, dy = 0.0, 0.0
            self.robot.heading += speed * 2.0
        elif "TURN_RIGHT" in gait:
            dx, dy = 0.0, 0.0
            self.robot.heading -= speed * 2.0
        else:
            dx, dy = 0.0, 0.0

        new_x = self.robot.pos.x + dx
        new_y = self.robot.pos.y + dy

        # Check boundary + collision
        if -4.5 < new_x < 4.5 and -4.5 < new_y < 4.5:
            self.robot.pos.x = new_x
            self.robot.pos.y = new_y
        else:
            reward -= 0.1   # Wall penalty

        collisions = self.physics.check_collision(self.robot, self.objects)
        if collisions:
            reward -= 0.2 * len(collisions)
            self.robot.collision_count += 1
            info["collisions"] = collisions

        # Head
        self.robot.head_pan  = float(action.get("head_pan",  self.robot.head_pan))
        self.robot.head_tilt = float(action.get("head_tilt", self.robot.head_tilt))

        # Gripper
        gesture = str(action.get("gesture", "OPEN"))
        if "GRASP" in gesture or "FIST" in gesture:
            self.robot.right_grip = 0.9
            # Try to pick up nearby object
            for obj in self.objects:
                if (obj.movable and obj.held_by is None and
                        self.physics.within_reach(self.robot, obj)):
                    obj.held_by = "right"
                    self.robot.right_held = obj.name
                    reward += 0.3
                    info["picked_up"] = obj.name
                    break
        elif "OPEN" in gesture:
            self.robot.right_grip = 0.0
            if self.robot.right_held:
                for obj in self.objects:
                    if obj.name == self.robot.right_held:
                        obj.held_by = None
                        obj.pos = Vec3(self.robot.pos.x + 0.5, self.robot.pos.y, 0.5)
                self.robot.right_held = None

        # Update held objects
        for obj in self.objects:
            if obj.held_by == "right":
                obj.pos = Vec3(self.robot.right_arm_pos.x,
                               self.robot.right_arm_pos.y,
                               self.robot.right_arm_pos.z)

        # Physics step
        for obj in self.objects:
            self.physics.step_object(obj, self.dt)

        # Move humans
        self._step_humans()

        # Task reward
        task_done, task_reward = self._check_task()
        reward += task_reward
        if task_done:
            info["task_complete"] = self._active_task["name"]
            self._next_task()

        # Battery
        if self.robot.battery < 0.1:
            reward -= 0.5
            info["low_battery"] = True

        # Episode limit
        if self._step >= 1000:
            done = True

        return reward, done, info

    def _step_humans(self):
        """Move simulated humans along their waypoints."""
        for human in self.humans:
            if not human.waypoints:
                continue
            wp = human.waypoints[human._wp_idx % len(human.waypoints)]
            dx = wp.x - human.pos.x
            dy = wp.y - human.pos.y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 0.3:
                human._wp_idx += 1
            else:
                human.pos.x += (dx / dist) * human.speed * self.dt
                human.pos.y += (dy / dist) * human.speed * self.dt
                human.heading = math.atan2(dy, dx)
            # Occasionally speak
            if random.random() < 0.005:
                human.is_speaking = True
                human.speech_text = random.choice([
                    "Hello robot!", "Can you help me?", "Good job!",
                    "Come here.", "What are you doing?",
                ])
            else:
                human.is_speaking = False

    def _check_task(self) -> Tuple[bool, float]:
        if not self._active_task:
            return False, 0.0
        task = self._active_task
        if task["name"] == "pick_up_cup1":
            if self.robot.right_held == "cup1":
                return True, 5.0
        elif task["name"] == "wave_to_person":
            for human in self.humans:
                if (human.name == "person_alice" and
                        self.robot.pos.dist(human.pos) < 2.0):
                    return True, 3.0
        return False, 0.0

    def _next_task(self):
        idx = self._tasks.index(self._active_task) if self._active_task in self._tasks else 0
        self._active_task = self._tasks[(idx + 1) % len(self._tasks)]
        print(f"[Sim] New task: {self._active_task['goal']}")

    # ── Rendering ─────────────────────────────────────────────────────

    def render_ascii(self) -> str:
        """ASCII top-down view of the world."""
        W, H = 40, 20
        grid = [[" "] * W for _ in range(H)]

        def to_grid(x, y):
            gx = int((x + 5) / 10 * W)
            gy = int((y + 5) / 10 * H)
            return max(0, min(W-1, gx)), max(0, min(H-1, gy))

        # Objects
        symbols = {"cup": "c", "chair": "C", "table": "T",
                   "obstacle": "X", "ball": "o"}
        for obj in self.objects:
            gx, gy = to_grid(obj.pos.x, obj.pos.y)
            grid[H-1-gy][gx] = symbols.get(obj.label, "?")

        # Humans
        for human in self.humans:
            gx, gy = to_grid(human.pos.x, human.pos.y)
            grid[H-1-gy][gx] = "H"

        # Robot
        gx, gy = to_grid(self.robot.pos.x, self.robot.pos.y)
        grid[H-1-gy][gx] = "R"

        lines = ["+" + "-"*W + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-"*W + "+")
        lines.append(f"Pos: ({self.robot.pos.x:.1f},{self.robot.pos.y:.1f}) "
                     f"Heading: {math.degrees(self.robot.heading):.0f}° "
                     f"Battery: {self.robot.battery*100:.0f}% "
                     f"Step: {self._step}")
        if self._active_task:
            lines.append(f"Task: {self._active_task['goal']}")
        return "\n".join(lines)
