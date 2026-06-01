"""
ChimeraRobot Embodied Memory
------------------------------
Memory systems specific to an embodied agent — very different from a chatbot's memory.

Four memory systems (inspired by neuroscience):

  1. Sensorimotor Buffer   — last N sensory frames (RAM, very fast)
                             Like the brain's sensory cortex short-term buffer
                             "What just happened a second ago?"

  2. Episodic Memory       — sequences of experiences (SQLite)
                             Like hippocampal episodic memory
                             "I picked up the red cup yesterday near the table"

  3. Spatial Memory        — a map of the environment (occupancy grid)
                             Like the brain's place cells + grid cells
                             "The door is 2m to my left, obstacle at (3,4)"

  4. Skill Library         — learned action sequences that worked (SQLite)
                             Like procedural memory / basal ganglia
                             "To pick up a cup: reach → grasp → lift works"

Together they give the robot continuity — it remembers what it did,
where things are, and how to do things it's done before.
"""

import json
import math
import os
import sqlite3
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────
#  1. Sensorimotor Buffer
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SensorySnapshot:
    """One moment of full sensory experience."""
    step:         int
    timestamp:    float
    world_emb:    List[float]          # Compact world embedding
    emotion:      str
    pain:         float
    detections:   List[str]            # Object labels seen
    is_speech:    bool
    action_taken: List[float]          # Motor action vector
    outcome:      str = "unknown"      # "success" | "failure" | "neutral"


class SensorimotorBuffer:
    """
    Fast ring buffer of recent sensory snapshots.
    Used for:
      - Short-term context for decisions
      - Detecting repeated patterns ("I keep hitting this wall")
      - Computing temporal derivatives ("am I moving toward or away?")
    """

    def __init__(self, capacity: int = 64):
        self._buf: Deque[SensorySnapshot] = deque(maxlen=capacity)

    def add(self, snap: SensorySnapshot):
        self._buf.append(snap)

    def last(self, n: int = 1) -> List[SensorySnapshot]:
        return list(self._buf)[-n:]

    def recent_pain_average(self, n: int = 10) -> float:
        recent = list(self._buf)[-n:]
        return float(np.mean([s.pain for s in recent])) if recent else 0.0

    def recent_emotions(self, n: int = 10) -> List[str]:
        return [s.emotion for s in list(self._buf)[-n:]]

    def dominant_recent_emotion(self, n: int = 10) -> str:
        emotions = self.recent_emotions(n)
        if not emotions:
            return "NEUTRAL"
        from collections import Counter
        return Counter(emotions).most_common(1)[0][0]

    def is_stuck(self, n: int = 8, threshold: float = 0.05) -> bool:
        """
        Detect if the robot is stuck (world embedding barely changing).
        """
        recent = list(self._buf)[-n:]
        if len(recent) < 2:
            return False
        embs = np.array([s.world_emb for s in recent])
        diffs = np.diff(embs, axis=0)
        return float(np.abs(diffs).mean()) < threshold

    def embedding_trajectory(self, n: int = 8) -> np.ndarray:
        """Return the last N world embeddings as a matrix (n, d)."""
        recent = list(self._buf)[-n:]
        return np.array([s.world_emb for s in recent]) if recent else np.array([])

    def __len__(self) -> int:
        return len(self._buf)


# ─────────────────────────────────────────────────────────────────────
#  2. Episodic Memory (SQLite)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    episode_id:   int
    start_time:   float
    end_time:     float
    summary:      str              # Natural language summary
    n_steps:      int
    avg_pain:     float
    dominant_emotion: str
    objects_seen: List[str]
    skills_used:  List[str]
    success:      bool
    location_x:   float = 0.0
    location_y:   float = 0.0


class EpisodicMemory:
    """
    SQLite-backed episodic memory.
    Stores summaries of past experience sequences.

    "I tried to pick up the cup. I moved forward, reached out,
     grasped it (grip_quality=0.8), lifted it successfully."
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time    REAL    NOT NULL,
        end_time      REAL    NOT NULL,
        summary       TEXT    NOT NULL,
        n_steps       INTEGER DEFAULT 0,
        avg_pain      REAL    DEFAULT 0.0,
        dominant_emotion TEXT DEFAULT 'NEUTRAL',
        objects_seen  TEXT    DEFAULT '[]',
        skills_used   TEXT    DEFAULT '[]',
        success       INTEGER DEFAULT 0,
        location_x    REAL    DEFAULT 0.0,
        location_y    REAL    DEFAULT 0.0
    );
    """

    def __init__(self, db_path: str = "robot_memory.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def save_episode(
        self,
        snapshots:   List[SensorySnapshot],
        summary:     str,
        skills_used: List[str] = None,
        success:     bool = True,
        location:    Tuple[float, float] = (0.0, 0.0),
    ) -> int:
        if not snapshots:
            return -1
        pains    = [s.pain for s in snapshots]
        emotions = [s.emotion for s in snapshots]
        objects  = list({obj for s in snapshots for obj in s.detections})
        from collections import Counter
        dom_emo  = Counter(emotions).most_common(1)[0][0]

        cur = self._conn.execute(
            """INSERT INTO episodes
               (start_time, end_time, summary, n_steps, avg_pain,
                dominant_emotion, objects_seen, skills_used, success, location_x, location_y)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (snapshots[0].timestamp, snapshots[-1].timestamp,
             summary, len(snapshots), float(np.mean(pains)),
             dom_emo, json.dumps(objects),
             json.dumps(skills_used or []), int(success),
             location[0], location[1]),
        )
        self._conn.commit()
        return cur.lastrowid

    def recall(self, query: str = "", limit: int = 5) -> List[Episode]:
        """Retrieve episodes matching a keyword query."""
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """SELECT episode_id, start_time, end_time, summary, n_steps,
                      avg_pain, dominant_emotion, objects_seen, skills_used,
                      success, location_x, location_y
               FROM episodes
               WHERE summary LIKE ? OR objects_seen LIKE ?
               ORDER BY end_time DESC LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        episodes = []
        for r in rows:
            episodes.append(Episode(
                episode_id=r[0], start_time=r[1], end_time=r[2],
                summary=r[3], n_steps=r[4], avg_pain=r[5],
                dominant_emotion=r[6],
                objects_seen=json.loads(r[7]),
                skills_used=json.loads(r[8]),
                success=bool(r[9]),
                location_x=r[10], location_y=r[11],
            ))
        return episodes

    def recent(self, n: int = 3) -> List[Episode]:
        return self.recall("", limit=n)

    def nearby_episodes(self, x: float, y: float, radius: float = 2.0) -> List[Episode]:
        """Find episodes that happened near a given location."""
        all_ep = self.recall("", limit=50)
        return [e for e in all_ep
                if math.sqrt((e.location_x - x)**2 + (e.location_y - y)**2) < radius]

    def success_rate(self) -> float:
        row = self._conn.execute("SELECT AVG(success) FROM episodes").fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    def stats(self) -> dict:
        n   = self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        sr  = self.success_rate()
        return {"total_episodes": n, "success_rate": round(sr, 3)}


# ─────────────────────────────────────────────────────────────────────
#  3. Spatial Memory (occupancy grid)
# ─────────────────────────────────────────────────────────────────────

class SpatialMemory:
    """
    2D occupancy grid map of the robot's environment.
    Inspired by the brain's place cells and grid cells.

    Grid:
      - FREE  (0.0)    : confirmed empty space
      - UNKNOWN (0.5)  : not yet explored
      - OCCUPIED (1.0) : obstacle / wall
      - INTERESTING (0.3): has interesting objects

    The robot's position is tracked and updated each step.
    """

    FREE       = 0.0
    UNKNOWN    = 0.5
    OCCUPIED   = 1.0
    INTERESTING = 0.3

    def __init__(
        self,
        width_m: float = 10.0,    # Physical size of the map (metres)
        height_m: float = 10.0,
        resolution: float = 0.1,  # metres per cell
    ):
        self.resolution = resolution
        self.width_cells  = int(width_m / resolution)
        self.height_cells = int(height_m / resolution)
        self._grid = np.full(
            (self.height_cells, self.width_cells), self.UNKNOWN, dtype=np.float32
        )
        # Object memory: {(cell_x, cell_y): label}
        self._objects: Dict[Tuple[int,int], str] = {}

        # Robot position (metres)
        self.robot_x = width_m / 2
        self.robot_y = height_m / 2
        self.robot_heading = 0.0   # Radians, 0 = facing +X

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        cx = int(x / self.resolution)
        cy = int(y / self.resolution)
        cx = max(0, min(self.width_cells  - 1, cx))
        cy = max(0, min(self.height_cells - 1, cy))
        return cx, cy

    def update_robot_position(self, dx: float, dy: float, dheading: float = 0.0):
        """Update robot's estimated position (dead reckoning)."""
        self.robot_x       += dx
        self.robot_y       += dy
        self.robot_heading  = (self.robot_heading + dheading) % (2 * math.pi)
        # Mark current cell as free
        cx, cy = self._world_to_cell(self.robot_x, self.robot_y)
        self._grid[cy, cx] = self.FREE

    def mark_obstacle(self, distance_m: float, angle_rad: float):
        """Mark an obstacle at (distance, angle) relative to robot."""
        wx = self.robot_x + distance_m * math.cos(self.robot_heading + angle_rad)
        wy = self.robot_y + distance_m * math.sin(self.robot_heading + angle_rad)
        cx, cy = self._world_to_cell(wx, wy)
        self._grid[cy, cx] = self.OCCUPIED

    def mark_object(self, label: str, distance_m: float, angle_rad: float):
        """Remember where an object was seen."""
        wx = self.robot_x + distance_m * math.cos(self.robot_heading + angle_rad)
        wy = self.robot_y + distance_m * math.sin(self.robot_heading + angle_rad)
        cx, cy = self._world_to_cell(wx, wy)
        self._grid[cy, cx] = self.INTERESTING
        self._objects[(cx, cy)] = label

    def find_object(self, label: str) -> Optional[Tuple[float, float]]:
        """
        Find where an object was last seen.
        Returns world (x, y) coordinates or None.
        """
        for (cx, cy), lbl in self._objects.items():
            if lbl == label:
                return cx * self.resolution, cy * self.resolution
        return None

    def nearest_unknown(self, radius_m: float = 3.0) -> Optional[Tuple[float, float]]:
        """
        Find the nearest unexplored cell — where to go to satisfy curiosity.
        """
        rx, ry = self._world_to_cell(self.robot_x, self.robot_y)
        r_cells = int(radius_m / self.resolution)
        best_dist = float("inf")
        best_pos  = None

        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                cx, cy = rx + dx, ry + dy
                if not (0 <= cx < self.width_cells and 0 <= cy < self.height_cells):
                    continue
                if self._grid[cy, cx] == self.UNKNOWN:
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos  = (cx * self.resolution, cy * self.resolution)
        return best_pos

    def exploration_coverage(self) -> float:
        """Fraction of the map that has been explored (not UNKNOWN)."""
        known = (self._grid != self.UNKNOWN).sum()
        return float(known) / self._grid.size

    def local_map(self, radius_cells: int = 10) -> np.ndarray:
        """Return the local map around the robot (2*radius × 2*radius)."""
        rx, ry = self._world_to_cell(self.robot_x, self.robot_y)
        r = radius_cells
        x0, x1 = max(0, rx-r), min(self.width_cells,  rx+r)
        y0, y1 = max(0, ry-r), min(self.height_cells, ry+r)
        return self._grid[y0:y1, x0:x1].copy()

    def is_blocked(self, dx: float, dy: float) -> bool:
        """Would moving (dx, dy) metres hit an obstacle?"""
        tx = self.robot_x + dx
        ty = self.robot_y + dy
        cx, cy = self._world_to_cell(tx, ty)
        return self._grid[cy, cx] == self.OCCUPIED


# ─────────────────────────────────────────────────────────────────────
#  4. Skill Library (procedural memory)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Skill:
    skill_id:     int
    name:         str
    description:  str
    action_seq:   List[dict]       # Sequence of BodyCommand-like dicts
    preconditions: List[str]       # What must be true to use this skill
    effects:      List[str]        # What this skill achieves
    success_count: int = 0
    failure_count: int = 0
    avg_duration: float = 0.0


class SkillLibrary:
    """
    Stores and retrieves learned action sequences (skills).
    Skills are generalizable procedures: "pick up object", "open door",
    "avoid obstacle", "wave hello".

    Skills can be:
    - Hardcoded (built-in behaviours)
    - Learned from successful episodes (reinforcement)
    - Composed from simpler skills
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS skills (
        skill_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        name           TEXT    NOT NULL UNIQUE,
        description    TEXT    DEFAULT '',
        action_seq     TEXT    DEFAULT '[]',
        preconditions  TEXT    DEFAULT '[]',
        effects        TEXT    DEFAULT '[]',
        success_count  INTEGER DEFAULT 0,
        failure_count  INTEGER DEFAULT 0,
        avg_duration   REAL    DEFAULT 0.0
    );
    """

    BUILTIN_SKILLS = [
        {
            "name": "wave_hello",
            "description": "Wave hand to greet a person",
            "action_seq": [
                {"gesture": "WAVE", "duration_s": 2.0},
            ],
            "preconditions": ["person_visible"],
            "effects": ["acknowledged_person"],
        },
        {
            "name": "pick_up_object",
            "description": "Reach and grasp a nearby object",
            "action_seq": [
                {"gesture": "OPEN",  "arm_z": 0.5, "duration_s": 0.5},
                {"gesture": "GRASP", "arm_z": 0.3, "duration_s": 0.5},
                {"gesture": "GRASP", "arm_z": 0.6, "duration_s": 0.5},
            ],
            "preconditions": ["object_visible", "object_reachable"],
            "effects": ["holding_object"],
        },
        {
            "name": "put_down_object",
            "description": "Lower and release held object",
            "action_seq": [
                {"gesture": "GRASP", "arm_z": 0.3, "duration_s": 0.5},
                {"gesture": "OPEN",  "arm_z": 0.3, "duration_s": 0.3},
                {"gesture": "OPEN",  "arm_z": 0.5, "duration_s": 0.3},
            ],
            "preconditions": ["holding_object"],
            "effects": ["object_placed"],
        },
        {
            "name": "avoid_obstacle",
            "description": "Step back and turn away from obstacle",
            "action_seq": [
                {"gait": "WALK_BACK",   "speed": 0.3, "duration_s": 1.0},
                {"gait": "TURN_RIGHT",  "speed": 0.3, "duration_s": 1.5},
                {"gait": "WALK_FORWARD","speed": 0.3, "duration_s": 1.0},
            ],
            "preconditions": ["obstacle_close"],
            "effects": ["obstacle_avoided"],
        },
        {
            "name": "express_joy",
            "description": "Show happiness through gestures",
            "action_seq": [
                {"gesture": "THUMBS_UP", "duration_s": 1.0},
                {"gesture": "WAVE",      "duration_s": 1.5},
            ],
            "preconditions": ["emotion_joy"],
            "effects": ["expressed_joy"],
        },
    ]

    def __init__(self, db_path: str = "robot_memory.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        self._seed_builtins()

    def _seed_builtins(self):
        for sk in self.BUILTIN_SKILLS:
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO skills (name, description, action_seq, preconditions, effects)"
                    " VALUES (?,?,?,?,?)",
                    (sk["name"], sk["description"],
                     json.dumps(sk["action_seq"]),
                     json.dumps(sk["preconditions"]),
                     json.dumps(sk["effects"])),
                )
            except Exception:
                pass
        self._conn.commit()

    def find_skill(self, query: str) -> Optional[Skill]:
        """Find a skill by name or description keyword."""
        row = self._conn.execute(
            "SELECT * FROM skills WHERE name LIKE ? OR description LIKE ? LIMIT 1",
            (f"%{query}%", f"%{query}%"),
        ).fetchone()
        return self._row_to_skill(row) if row else None

    def skills_for_situation(self, conditions: List[str]) -> List[Skill]:
        """Return all skills whose preconditions match the current situation."""
        all_skills = self._conn.execute("SELECT * FROM skills").fetchall()
        matching   = []
        for row in all_skills:
            sk = self._row_to_skill(row)
            precond = sk.preconditions
            if any(c in conditions for c in precond) or not precond:
                matching.append(sk)
        return sorted(matching, key=lambda s: s.success_count, reverse=True)

    def record_outcome(self, skill_name: str, success: bool, duration_s: float):
        if success:
            self._conn.execute(
                "UPDATE skills SET success_count = success_count + 1,"
                " avg_duration = (avg_duration + ?) / 2"
                " WHERE name = ?", (duration_s, skill_name)
            )
        else:
            self._conn.execute(
                "UPDATE skills SET failure_count = failure_count + 1 WHERE name = ?",
                (skill_name,)
            )
        self._conn.commit()

    def learn_skill(
        self,
        name: str,
        description: str,
        action_seq: List[dict],
        preconditions: List[str],
        effects: List[str],
    ) -> int:
        cur = self._conn.execute(
            "INSERT OR REPLACE INTO skills (name, description, action_seq, preconditions, effects)"
            " VALUES (?,?,?,?,?)",
            (name, description, json.dumps(action_seq),
             json.dumps(preconditions), json.dumps(effects)),
        )
        self._conn.commit()
        print(f"[Skills] Learned new skill: '{name}'")
        return cur.lastrowid

    @staticmethod
    def _row_to_skill(row) -> Skill:
        return Skill(
            skill_id=row[0], name=row[1], description=row[2],
            action_seq=json.loads(row[3]),
            preconditions=json.loads(row[4]),
            effects=json.loads(row[5]),
            success_count=row[6], failure_count=row[7], avg_duration=row[8],
        )

    def stats(self) -> dict:
        n = self._conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        return {"total_skills": n}


# ─────────────────────────────────────────────────────────────────────
#  Unified embodied memory
# ─────────────────────────────────────────────────────────────────────

class EmbodiedMemory:
    """
    Facade combining all four memory systems.
    This is what the robot_brain.py accesses for all memory operations.
    """

    def __init__(self, db_path: str = "robot_memory.db"):
        self.sensorimotor = SensorimotorBuffer(capacity=128)
        self.episodic     = EpisodicMemory(db_path)
        self.spatial      = SpatialMemory()
        self.skills       = SkillLibrary(db_path)
        self._episode_buf: List[SensorySnapshot] = []
        self._episode_start = time.time()

    def record_step(self, snapshot: SensorySnapshot):
        """Record one timestep across all memory systems."""
        self.sensorimotor.add(snapshot)
        self._episode_buf.append(snapshot)

    def close_episode(self, summary: str, success: bool = True,
                      skills_used: List[str] = None):
        """Commit current episode to episodic memory and reset buffer."""
        self.episodic.save_episode(
            self._episode_buf, summary,
            skills_used=skills_used, success=success,
            location=(self.spatial.robot_x, self.spatial.robot_y),
        )
        self._episode_buf.clear()
        self._episode_start = time.time()

    def stats(self) -> dict:
        return {
            "sensorimotor_frames": len(self.sensorimotor),
            "is_stuck": self.sensorimotor.is_stuck(),
            "map_coverage_%": round(self.spatial.exploration_coverage() * 100, 1),
            **self.episodic.stats(),
            **self.skills.stats(),
        }
