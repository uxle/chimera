"""
ChimeraRobot Multimodal Fusion
--------------------------------
Fuses ALL sensory streams into a single unified embedding
that the language brain (ChimeraLLM) can reason about.

This is the key integration layer — it answers the question:
"What is the world like right now, in all modalities, as one vector?"

Architecture:
  Vision   (512) ─┐
  Audio    (256) ─┤→ Cross-Modal Attention → Gated Fusion → 512-dim embedding
  Touch    (64)  ─┤
  Proprio  (128) ─┘
  Emotion  (3)   ─┘

Cross-modal attention lets modalities "ask questions" of each other:
  "Vision sees a cup — does touch confirm I'm holding it?"
  "Audio hears crying — does vision confirm a face?"
This produces a richer, more coherent world model than simple concatenation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_config import BrainConfig


# ─────────────────────────────────────────────────────────────────────
#  Unified sensory frame
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SensoryFrame:
    """All sensory inputs at one timestep."""
    vision_emb:   Optional[torch.Tensor] = None   # (vision_feature_dim,)
    audio_emb:    Optional[torch.Tensor] = None   # (audio_feature_dim,)
    touch_emb:    Optional[torch.Tensor] = None   # (touch_feature_dim,)
    proprio_emb:  Optional[torch.Tensor] = None   # (proprio_feature_dim,)
    emotion_pad:  Optional[torch.Tensor] = None   # (3,) PAD vector
    pain_level:   float = 0.0
    is_speech:    bool  = False
    motion_mag:   float = 0.0
    n_detections: int   = 0


# ─────────────────────────────────────────────────────────────────────
#  Per-modality projectors (align all to d_model)
# ─────────────────────────────────────────────────────────────────────

class ModalityProjector(nn.Module):
    """Projects a modality embedding to d_model with LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (projected, gate_weight)"""
        g = self.gate(x)
        return self.norm(self.proj(x)), g


# ─────────────────────────────────────────────────────────────────────
#  Cross-modal attention
# ─────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Each modality attends to all other modalities.
    This allows cross-modal reasoning:
    "I see a ball AND I'm touching it" → stronger representation of "holding ball"

    Architecture: each modality token is a query; all tokens are keys+values.
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.0, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:  tokens: (B, n_modalities, d_model)
        Returns: (B, n_modalities, d_model) — each token enriched by others
        """
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + out)


# ─────────────────────────────────────────────────────────────────────
#  Temporal buffer (short-term sensory memory)
# ─────────────────────────────────────────────────────────────────────

class TemporalBuffer(nn.Module):
    """
    Maintains a short buffer of recent fused embeddings.
    Encodes temporal patterns (motion, change over time) using a GRU.
    This gives the robot a short-term "working memory" of the last N frames.
    """

    def __init__(self, d_model: int, buffer_len: int = 8):
        super().__init__()
        self.buffer_len = buffer_len
        self.gru = nn.GRU(
            input_size=d_model, hidden_size=d_model,
            num_layers=2, batch_first=True,
        )
        self._hidden: Optional[torch.Tensor] = None
        self._buffer: list = []

    def update(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:  emb: (d_model,) current frame embedding
        Returns: (d_model,) temporally-enriched embedding
        """
        emb_b = emb.unsqueeze(0).unsqueeze(0)   # (1, 1, d_model)
        out, self._hidden = self.gru(emb_b, self._hidden)
        self._buffer.append(emb)
        if len(self._buffer) > self.buffer_len:
            self._buffer.pop(0)
        return out.squeeze(0).squeeze(0)

    def reset(self):
        self._hidden = None
        self._buffer.clear()


# ─────────────────────────────────────────────────────────────────────
#  Scalar context encoder
# ─────────────────────────────────────────────────────────────────────

class ScalarContextEncoder(nn.Module):
    """
    Encodes scalar signals (pain, motion, n_detections, emotion) into the
    same d_model space so they can contribute to the fusion.
    These are important but tiny signals that mustn't be ignored.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Inputs: pain(1) + motion(1) + n_detections(1) + is_speech(1) + PAD(3) = 7
        self.enc = nn.Sequential(
            nn.Linear(7, 64),
            nn.SiLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, frame: SensoryFrame) -> torch.Tensor:
        """Returns (d_model,) scalar context embedding."""
        pad = frame.emotion_pad if frame.emotion_pad is not None else torch.zeros(3)
        scalars = torch.tensor([
            frame.pain_level,
            frame.motion_mag,
            min(frame.n_detections / 5.0, 1.0),   # Normalise
            float(frame.is_speech),
            float(pad[0]), float(pad[1]), float(pad[2]),
        ], dtype=torch.float32)
        return self.enc(scalars)


# ─────────────────────────────────────────────────────────────────────
#  Main fusion module
# ─────────────────────────────────────────────────────────────────────

class MultimodalFusion(nn.Module):
    """
    Complete multimodal fusion pipeline.

    Takes a SensoryFrame → returns a single (d_model,) embedding
    that encodes the full state of the world from ALL senses simultaneously.

    This embedding is what the ChimeraLLM brain processes to decide:
    - What to say
    - Where to move
    - How to feel
    - What to pay attention to
    """

    N_MODALITIES = 5   # vision, audio, touch, proprio, scalar_context

    def __init__(self, config: BrainConfig):
        super().__init__()
        d = config.d_model

        # Per-modality projectors
        self.vision_proj  = ModalityProjector(config.vision_feature_dim,  d)
        self.audio_proj   = ModalityProjector(config.audio_feature_dim,   d)
        self.touch_proj   = ModalityProjector(config.touch_feature_dim,   d)
        self.proprio_proj = ModalityProjector(config.proprio_feature_dim, d)
        self.scalar_enc   = ScalarContextEncoder(d)

        # Modality-type embeddings (learnable "which sense am I from?")
        self.modality_emb = nn.Embedding(self.N_MODALITIES, d)

        # Cross-modal attention (3 rounds)
        self.cross_attn = nn.Sequential(
            CrossModalAttention(d, n_heads=4),
            CrossModalAttention(d, n_heads=4),
            CrossModalAttention(d, n_heads=4),
        )

        # Gated aggregation: weighted sum of modality tokens
        self.agg_gate = nn.Sequential(
            nn.Linear(d * self.N_MODALITIES, self.N_MODALITIES),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.LayerNorm(d),
        )

        # Temporal buffer
        self.temporal = TemporalBuffer(d, buffer_len=8)

        # Missing-modality fallback tokens (learned)
        self.null_vision  = nn.Parameter(torch.randn(d) * 0.01)
        self.null_audio   = nn.Parameter(torch.randn(d) * 0.01)
        self.null_touch   = nn.Parameter(torch.randn(d) * 0.01)
        self.null_proprio = nn.Parameter(torch.randn(d) * 0.01)

    def _project(
        self,
        emb: Optional[torch.Tensor],
        projector: ModalityProjector,
        null_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project modality or substitute learned null token if missing."""
        if emb is None:
            return null_token, torch.tensor(0.0)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return projector(emb)

    def forward(self, frame: SensoryFrame) -> torch.Tensor:
        """
        Args:  frame: SensoryFrame with all available modalities
        Returns: (d_model,) unified world embedding
        """
        # Project each modality
        v_emb, v_gate  = self._project(frame.vision_emb,  self.vision_proj,  self.null_vision)
        a_emb, a_gate  = self._project(frame.audio_emb,   self.audio_proj,   self.null_audio)
        t_emb, t_gate  = self._project(frame.touch_emb,   self.touch_proj,   self.null_touch)
        p_emb, p_gate  = self._project(frame.proprio_emb, self.proprio_proj, self.null_proprio)

        # Scalar context
        sc_emb = self.scalar_enc(frame)

        # Squeeze all to (d_model,) 1D tensors
        tokens = torch.stack([
            v_emb.squeeze(0),
            a_emb.squeeze(0),
            t_emb.squeeze(0),
            p_emb.squeeze(0),
            sc_emb,
        ], dim=0).unsqueeze(0)   # (1, 5, d_model)

        # Add modality-type embeddings
        mod_ids  = torch.arange(self.N_MODALITIES, device=tokens.device)
        mod_embs = self.modality_emb(mod_ids).unsqueeze(0)  # (1, 5, d)
        tokens   = tokens + mod_embs

        # Cross-modal attention
        for layer in self.cross_attn:
            tokens = layer(tokens)

        # Gated aggregation
        flat  = tokens.squeeze(0).flatten()         # (5*d,)
        gates = self.agg_gate(flat)                  # (5,) softmax weights
        agg   = (tokens.squeeze(0) * gates.unsqueeze(-1)).sum(0)  # (d,)

        # Final projection
        out = self.out_proj(agg)

        # Temporal enrichment
        out = self.temporal.update(out)

        return out

    def reset_temporal(self):
        """Call when starting a new episode."""
        self.temporal.reset()

    def describe_frame(self, frame: SensoryFrame) -> str:
        """Produce a human-readable description of the sensory frame."""
        parts = []
        if frame.vision_emb is not None:
            parts.append(f"👁 vision(motion={frame.motion_mag:.2f}, dets={frame.n_detections})")
        if frame.audio_emb is not None:
            speech = "🗣speech" if frame.is_speech else "🔇quiet"
            parts.append(f"👂 {speech}")
        if frame.touch_emb is not None:
            parts.append(f"✋ touch(pain={frame.pain_level:.2f})")
        if frame.proprio_emb is not None:
            parts.append("⚖ proprioception")
        if frame.emotion_pad is not None:
            p, a, d = frame.emotion_pad.tolist()
            parts.append(f"💭 PAD({p:+.2f},{a:+.2f},{d:+.2f})")
        return " | ".join(parts) if parts else "No sensory input"
