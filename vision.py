"""
ChimeraRobot Vision System
---------------------------
Processes camera frames into rich feature embeddings the brain can use.

Pipeline:
  Raw frame (H×W×3)
    → Preprocessing (normalize, resize)
    → PatchEncoder (ViT-style: divide into patches, embed each)
    → SpatialAttention (attend across patches)
    → SceneEmbedding (512-dim summary vector)
    → ObjectDetector (bounding boxes + labels from visual vocab)
    → MotionDetector (optical flow → movement signals)
    → DepthEstimator (monocular depth — how far things are)

All networks are from-scratch (no pretrained weights required).
Falls back to MockCamera in simulation mode.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_config import CameraConfig


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]   # (x1, y1, x2, y2) normalised 0–1
    depth_m: Optional[float] = None           # Estimated distance in metres


@dataclass
class VisualScene:
    frame:           np.ndarray                # Raw H×W×3 uint8
    embedding:       torch.Tensor             # (feature_dim,) scene summary
    patch_tokens:    torch.Tensor             # (n_patches, patch_embed_dim) spatial tokens
    detections:      List[Detection]
    motion_magnitude: float                   # 0 = still, 1 = lots of movement
    depth_map:       Optional[torch.Tensor]   # (H/8, W/8) relative depth
    timestamp:       float = 0.0


# ─────────────────────────────────────────────────────────────────────
#  Patch encoder (ViT-style)
# ─────────────────────────────────────────────────────────────────────

class PatchEncoder(nn.Module):
    """
    Splits image into non-overlapping patches and embeds each with a linear layer.
    E.g. 224×224 image with 16×16 patches → 196 patch tokens.

    Each patch token carries LOCAL visual information.
    A class token (CLS) aggregates GLOBAL scene information.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 256):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        self.embed_dim  = embed_dim

        # Linear projection of flattened patches
        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_emb   = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:  x: (B, C, H, W)
        Returns:
            cls_out:    (B, embed_dim) — global scene token
            patch_out:  (B, n_patches, embed_dim) — spatial patch tokens
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Split into patches: (B, n_patches, patch_dim)
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, -1, C * p * p)

        tokens = self.proj(patches)              # (B, n_patches, embed_dim)
        cls    = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, n_patches+1, embed_dim)
        tokens = tokens + self.pos_emb
        tokens = self.norm(tokens)

        return tokens[:, 0], tokens[:, 1:]   # cls, patch_tokens


class SpatialAttentionEncoder(nn.Module):
    """
    Self-attention over patch tokens to let patches communicate
    (distant regions can influence each other).
    Produces a rich spatial representation.
    """

    def __init__(self, embed_dim: int = 256, n_heads: int = 4, n_layers: int = 3,
                 output_dim: int = 512):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0, batch_first=True,
            norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj_out = nn.Linear(embed_dim, output_dim)

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            global_emb:   (B, output_dim) pooled scene embedding
            spatial_emb:  (B, n_patches, output_dim) per-patch embeddings
        """
        out = self.encoder(patch_tokens)                    # (B, n_patches, embed_dim)
        spatial = self.proj_out(out)                        # (B, n_patches, output_dim)
        global_emb = spatial.mean(dim=1)                    # (B, output_dim)
        return global_emb, spatial


# ─────────────────────────────────────────────────────────────────────
#  Object detector (lightweight, from-scratch)
# ─────────────────────────────────────────────────────────────────────

# Visual vocabulary — what the robot knows how to look for
VISUAL_VOCAB = [
    "person", "hand", "face", "robot", "chair", "table", "door",
    "bottle", "cup", "book", "phone", "ball", "box", "bag",
    "obstacle", "floor", "wall", "light", "window", "unknown",
]
N_VISUAL_CLASSES = len(VISUAL_VOCAB)


class LightweightDetector(nn.Module):
    """
    Single-shot object detector operating on patch tokens.
    Each patch predicts: is there an object? what class? where exactly?
    Not state-of-the-art accuracy but runs fast on CPU.
    """

    def __init__(self, patch_dim: int = 512, n_classes: int = N_VISUAL_CLASSES,
                 n_anchors: int = 3):
        super().__init__()
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        out_per_anchor = 5 + n_classes   # (objectness, x, y, w, h, *class_probs)

        self.head = nn.Sequential(
            nn.Linear(patch_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_anchors * out_per_anchor),
        )

    def forward(self, spatial_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:  spatial_emb: (B, n_patches, patch_dim)
        Returns: (B, n_patches, n_anchors, 5+n_classes)
        """
        B, P, D = spatial_emb.shape
        out = self.head(spatial_emb)    # (B, P, n_anchors*(5+n_classes))
        out = out.view(B, P, self.n_anchors, 5 + self.n_classes)
        return out

    def decode(
        self, pred: torch.Tensor, n_patches_per_side: int,
        conf_threshold: float = 0.4
    ) -> List[Detection]:
        """Decode raw predictions into Detection objects."""
        B, P, A, _ = pred.shape
        detections = []
        objectness = torch.sigmoid(pred[0, :, :, 0])    # (P, A)
        classes    = F.softmax(pred[0, :, :, 5:], dim=-1)
        boxes      = torch.sigmoid(pred[0, :, :, 1:5])

        for p_idx in range(P):
            row = p_idx // n_patches_per_side
            col = p_idx %  n_patches_per_side
            for a_idx in range(A):
                conf = float(objectness[p_idx, a_idx])
                if conf < conf_threshold:
                    continue
                cls_probs = classes[p_idx, a_idx]
                cls_id    = int(cls_probs.argmax())
                cls_conf  = float(cls_probs[cls_id]) * conf

                # Decode box relative to patch position
                bx = (col + float(boxes[p_idx, a_idx, 0])) / n_patches_per_side
                by = (row + float(boxes[p_idx, a_idx, 1])) / n_patches_per_side
                bw = float(boxes[p_idx, a_idx, 2])
                bh = float(boxes[p_idx, a_idx, 3])

                detections.append(Detection(
                    label=VISUAL_VOCAB[cls_id],
                    confidence=round(cls_conf, 3),
                    bbox=(max(0, bx - bw/2), max(0, by - bh/2),
                          min(1, bx + bw/2), min(1, by + bh/2)),
                ))
        return detections


# ─────────────────────────────────────────────────────────────────────
#  Monocular depth estimator
# ─────────────────────────────────────────────────────────────────────

class DepthEstimator(nn.Module):
    """
    Estimates relative depth from a single RGB frame.
    Output is a coarse depth map (H/8 × W/8).
    Values are relative (0 = close, 1 = far) — not absolute metres.
    Can be calibrated against known distances.
    """

    def __init__(self, patch_dim: int = 512, n_patches_per_side: int = 14):
        super().__init__()
        self.n = n_patches_per_side
        self.decoder = nn.Sequential(
            nn.Linear(patch_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, spatial_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:  spatial_emb: (B, n_patches, patch_dim)
        Returns: (B, n, n) depth map
        """
        depth = self.decoder(spatial_emb).squeeze(-1)   # (B, n_patches)
        B, P = depth.shape
        return depth.view(B, self.n, self.n)


# ─────────────────────────────────────────────────────────────────────
#  Motion detector (frame differencing)
# ─────────────────────────────────────────────────────────────────────

class MotionDetector:
    """
    Detects motion between consecutive frames using frame differencing.
    Simple but fast — good enough for "something moved" detection.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold  = threshold
        self._prev_gray: Optional[np.ndarray] = None

    def process(self, frame: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Args:   frame: (H, W, 3) uint8
        Returns: (motion_magnitude, motion_map)
        """
        gray = frame.mean(axis=2).astype(np.float32) / 255.0
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0, np.zeros_like(gray)

        diff = np.abs(gray - self._prev_gray)
        motion_map = (diff > self.threshold).astype(np.float32)
        magnitude  = float(motion_map.mean())
        self._prev_gray = gray
        return magnitude, motion_map

    def reset(self):
        self._prev_gray = None


# ─────────────────────────────────────────────────────────────────────
#  Full vision encoder
# ─────────────────────────────────────────────────────────────────────

class VisionEncoder(nn.Module):
    """
    Full vision processing pipeline.
    Input:  raw RGB frame as numpy array
    Output: VisualScene with embeddings + detections + depth + motion
    """

    def __init__(self, config: CameraConfig):
        super().__init__()
        self.config = config
        img_size    = config.width           # Assumes square (224×224)
        patch_size  = config.patch_size
        embed_dim   = 256
        self.output_dim = config.feature_dim

        n_patches_side  = img_size // patch_size

        self.patch_encoder    = PatchEncoder(img_size, patch_size, 3, embed_dim)
        self.spatial_attn     = SpatialAttentionEncoder(embed_dim, n_heads=4,
                                                         n_layers=3,
                                                         output_dim=self.output_dim)
        self.detector         = LightweightDetector(self.output_dim)
        self.depth_estimator  = DepthEstimator(self.output_dim, n_patches_side)
        self.motion_detector  = MotionDetector()
        self.n_patches_side   = n_patches_side

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """uint8 H×W×3 → float32 tensor 1×3×H×W, normalised"""
        img = frame.astype(np.float32) / 255.0
        if img.shape[:2] != (self.config.height, self.config.width):
            # Simple resize via interpolation
            from PIL import Image
            img_pil = Image.fromarray(frame)
            img_pil = img_pil.resize((self.config.width, self.config.height))
            img = np.array(img_pil).astype(np.float32) / 255.0
        # Normalise (ImageNet-style)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    @torch.no_grad()
    def encode(self, frame: np.ndarray) -> VisualScene:
        """Full pipeline: frame → VisualScene"""
        self.eval()

        # Motion
        motion_mag, _ = self.motion_detector.process(frame)

        # Preprocess
        x = self.preprocess(frame)

        # Patch encoding
        cls_token, patch_tokens = self.patch_encoder(x)

        # Spatial attention
        global_emb, spatial_emb = self.spatial_attn(patch_tokens)

        # Detection
        det_pred   = self.detector(spatial_emb)
        detections = self.detector.decode(det_pred, self.n_patches_side)

        # Depth
        depth_map = self.depth_estimator(spatial_emb).squeeze(0)

        return VisualScene(
            frame=frame,
            embedding=global_emb.squeeze(0),
            patch_tokens=spatial_emb.squeeze(0),
            detections=detections,
            motion_magnitude=motion_mag,
            depth_map=depth_map,
            timestamp=time.time(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trainable forward: returns (global_emb, spatial_emb)."""
        cls_token, patch_tokens  = self.patch_encoder(x)
        global_emb, spatial_emb  = self.spatial_attn(patch_tokens)
        return global_emb, spatial_emb


# ─────────────────────────────────────────────────────────────────────
#  Hardware / simulation interface
# ─────────────────────────────────────────────────────────────────────

class CameraInterface:
    """Abstract camera interface. Real and mock share this API."""

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self):
        pass


class RealCamera(CameraInterface):
    """OpenCV-based real camera driver."""

    def __init__(self, config: CameraConfig):
        try:
            import cv2
            self._cap = cv2.VideoCapture(config.device_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            self._cap.set(cv2.CAP_PROP_FPS, config.fps)
            self._cv2 = cv2
        except ImportError:
            raise ImportError("pip install opencv-python")

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)

    def release(self):
        self._cap.release()


class MockCamera(CameraInterface):
    """
    Generates synthetic frames for simulation.
    Produces structured noise + moving blobs to simulate a real scene.
    """

    def __init__(self, config: CameraConfig):
        self.config = config
        self._t = 0

    def read(self) -> np.ndarray:
        H, W = self.config.height, self.config.width
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Sky gradient
        frame[:H//2, :] = [100, 150, 200]
        # Floor
        frame[H//2:, :] = [80, 70, 60]

        # Moving blob (simulated object)
        cx = int(W//2 + W//3 * math.sin(self._t * 0.05))
        cy = int(H//3 + H//6 * math.cos(self._t * 0.03))
        r  = 20
        y, x = np.ogrid[:H, :W]
        mask = (x - cx)**2 + (y - cy)**2 < r**2
        frame[mask] = [255, 180, 50]  # Orange blob = "person"

        self._t += 1
        noise = np.random.randint(0, 15, (H, W, 3), dtype=np.uint8)
        return np.clip(frame.astype(np.int32) + noise, 0, 255).astype(np.uint8)
