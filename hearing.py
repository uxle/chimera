"""
ChimeraRobot Hearing System
----------------------------
Processes microphone input into audio embeddings the brain can understand.

Pipeline:
  Raw audio (PCM samples)
    → Pre-emphasis filter (boost high freqs)
    → MFCC extraction (how human ears perceive sound)
    → Temporal conv encoder (pattern over time)
    → Audio embedding (256-dim)
    → Sound classifier (speech / music / noise / silence / alarm)
    → Speech activity detector (is someone talking?)
    → Speech transcription hook (STT integration)

All signal processing is from-scratch (numpy only).
Neural encoding uses a lightweight 1D CNN + attention.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robot_config import AudioConfig


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

SOUND_CLASSES = [
    "silence", "speech", "music", "noise",
    "alarm", "crash", "laughter", "crying", "applause", "unknown",
]
N_SOUND_CLASSES = len(SOUND_CLASSES)


@dataclass
class AudioScene:
    raw_audio:      np.ndarray        # (n_samples,) float32 -1..1
    mfcc:           np.ndarray        # (n_mfcc, n_frames)
    embedding:      torch.Tensor      # (feature_dim,)
    is_speech:      bool
    speech_prob:    float             # 0.0–1.0
    sound_class:    str               # dominant sound type
    sound_probs:    dict              # {class: prob}
    loudness_db:    float
    timestamp:      float = 0.0


# ─────────────────────────────────────────────────────────────────────
#  MFCC feature extraction (from scratch, numpy only)
# ─────────────────────────────────────────────────────────────────────

class MFCCExtractor:
    """
    Mel-Frequency Cepstral Coefficients — the standard audio feature for speech.
    Mimics the logarithmic frequency response of the human cochlea.
    """

    def __init__(self, config: AudioConfig):
        self.sr         = config.sample_rate
        self.n_mfcc     = config.n_mfcc
        self.n_fft      = config.n_fft
        self.hop        = config.hop_length
        self.n_mels     = 64
        self._mel_bank  = self._build_mel_filterbank()

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        """Build triangular mel filterbank: (n_mels, n_fft//2+1)"""
        lo_mel = self._hz_to_mel(0)
        hi_mel = self._hz_to_mel(self.sr / 2)
        mel_pts = np.linspace(lo_mel, hi_mel, self.n_mels + 2)
        hz_pts  = np.array([self._mel_to_hz(m) for m in mel_pts])
        bin_pts = np.floor((self.n_fft + 1) * hz_pts / self.sr).astype(int)

        fbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_pts[m - 1]
            f_m       = bin_pts[m]
            f_m_plus  = bin_pts[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m-1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)
            for k in range(f_m, f_m_plus):
                fbank[m-1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)
        return fbank

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Args:  audio: (n_samples,) float32, range -1..1
        Returns: mfcc: (n_mfcc, n_frames) float32
        """
        # Pre-emphasis (boost high frequencies)
        emphasised = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # Framing
        frames = []
        for start in range(0, len(emphasised) - self.n_fft, self.hop):
            frame = emphasised[start : start + self.n_fft]
            frames.append(frame)
        if not frames:
            return np.zeros((self.n_mfcc, 1))

        frames = np.array(frames)  # (n_frames, n_fft)

        # Window (Hamming)
        window = np.hamming(self.n_fft)
        frames = frames * window

        # FFT magnitude spectrum
        mag = np.abs(np.fft.rfft(frames, n=self.n_fft))  # (n_frames, n_fft//2+1)

        # Mel filterbank
        mel_spec = np.dot(mag, self._mel_bank.T)  # (n_frames, n_mels)
        mel_spec = np.maximum(mel_spec, 1e-10)
        log_mel  = 20 * np.log10(mel_spec)        # log scale in dB

        # DCT (discrete cosine transform) → MFCCs
        n_frames, n_mels = log_mel.shape
        mfcc = np.zeros((n_frames, self.n_mfcc))
        for n in range(self.n_mfcc):
            for m in range(n_mels):
                mfcc[:, n] += log_mel[:, m] * math.cos(math.pi * n * (2 * m + 1) / (2 * n_mels))

        return mfcc.T.astype(np.float32)  # (n_mfcc, n_frames)


# ─────────────────────────────────────────────────────────────────────
#  Temporal audio encoder (1D CNN + attention)
# ─────────────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """
    Encodes MFCC feature sequences into a fixed-size embedding.
    Architecture: 1D depthwise-separable CNN → pooling → linear

    1D CNN captures short-term temporal patterns (phonemes, beats).
    Global average pooling collapses time → fixed-size vector.
    """

    def __init__(self, n_mfcc: int = 40, output_dim: int = 256):
        super().__init__()

        # Depthwise-separable conv blocks (efficient for mobile/robot)
        self.conv_layers = nn.Sequential(
            self._dw_block(n_mfcc, 128, kernel=5),
            self._dw_block(128,    128, kernel=5),
            self._dw_block(128,    256, kernel=3),
            self._dw_block(256,    256, kernel=3),
        )

        # Temporal attention: weight which frames matter most
        self.attn_query = nn.Linear(256, 64)
        self.attn_key   = nn.Linear(256, 64)

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Sound classifier
        self.classifier = nn.Linear(output_dim, N_SOUND_CLASSES)

        # Speech activity detector
        self.vad = nn.Linear(output_dim, 2)   # binary: speech / no-speech

    @staticmethod
    def _dw_block(in_ch: int, out_ch: int, kernel: int) -> nn.Module:
        """Depthwise-separable conv block."""
        return nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel, padding=kernel//2, groups=in_ch),
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
        )

    def forward(self, mfcc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:  mfcc: (B, n_mfcc, T)
        Returns:
            embedding:    (B, output_dim)
            sound_logits: (B, N_SOUND_CLASSES)
            vad_logits:   (B, 2)
        """
        x = self.conv_layers(mfcc)       # (B, 256, T')

        # Temporal attention pooling
        x_t = x.permute(0, 2, 1)        # (B, T', 256)
        q = self.attn_query(x_t)         # (B, T', 64)
        k = self.attn_key(x_t)
        scores = torch.softmax((q * k).sum(-1, keepdim=True) / 8.0, dim=1)  # (B, T', 1)
        pooled = (x_t * scores).sum(1)   # (B, 256)

        emb          = self.proj(pooled)
        sound_logits = self.classifier(emb)
        vad_logits   = self.vad(emb)
        return emb, sound_logits, vad_logits


# ─────────────────────────────────────────────────────────────────────
#  Loudness meter
# ─────────────────────────────────────────────────────────────────────

def compute_loudness_db(audio: np.ndarray, ref: float = 1.0) -> float:
    """RMS loudness in dBFS."""
    rms = math.sqrt(float(np.mean(audio ** 2)) + 1e-10)
    return 20 * math.log10(rms / ref)


# ─────────────────────────────────────────────────────────────────────
#  Full hearing system
# ─────────────────────────────────────────────────────────────────────

class HearingSystem:
    """
    Complete hearing pipeline: raw audio → AudioScene.
    """

    def __init__(self, config: AudioConfig):
        self.config = config
        self.mfcc_extractor = MFCCExtractor(config)
        self.encoder        = AudioEncoder(config.n_mfcc, config.feature_dim)
        self._audio_buffer: List[np.ndarray] = []
        self._buffer_samples = int(config.buffer_seconds * config.sample_rate)

    def add_samples(self, samples: np.ndarray):
        """Feed raw PCM samples into the buffer."""
        self._audio_buffer.append(samples)
        total = sum(len(s) for s in self._audio_buffer)
        while total > self._buffer_samples * 2:
            removed = self._audio_buffer.pop(0)
            total -= len(removed)

    @torch.no_grad()
    def process(self) -> AudioScene:
        """Process current buffer → AudioScene."""
        self.encoder.eval()

        if self._audio_buffer:
            audio = np.concatenate(self._audio_buffer)[-self._buffer_samples:]
        else:
            audio = np.zeros(self._buffer_samples, dtype=np.float32)

        audio = audio.astype(np.float32)
        loudness = compute_loudness_db(audio)

        mfcc = self.mfcc_extractor.extract(audio)      # (n_mfcc, T)

        # Pad / trim to fixed length
        target_len = 128
        if mfcc.shape[1] < target_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :target_len]

        mfcc_t = torch.from_numpy(mfcc).unsqueeze(0)   # (1, n_mfcc, T)

        emb, sound_logits, vad_logits = self.encoder(mfcc_t)

        sound_probs = F.softmax(sound_logits, dim=-1).squeeze(0).tolist()
        sound_dict  = {cls: round(p, 3) for cls, p in zip(SOUND_CLASSES, sound_probs)}
        dominant    = SOUND_CLASSES[int(sound_logits.argmax(-1))]

        vad_probs   = F.softmax(vad_logits, dim=-1).squeeze(0)
        speech_prob = float(vad_probs[1])

        return AudioScene(
            raw_audio=audio,
            mfcc=mfcc,
            embedding=emb.squeeze(0),
            is_speech=(speech_prob > 0.5),
            speech_prob=round(speech_prob, 3),
            sound_class=dominant,
            sound_probs=sound_dict,
            loudness_db=round(loudness, 1),
            timestamp=time.time(),
        )


# ─────────────────────────────────────────────────────────────────────
#  Hardware / mock interface
# ─────────────────────────────────────────────────────────────────────

class MicrophoneInterface:
    def read_chunk(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self):
        pass


class RealMicrophone(MicrophoneInterface):
    """PyAudio-based real microphone."""

    def __init__(self, config: AudioConfig):
        try:
            import pyaudio
            self._pa     = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=config.channels,
                rate=config.sample_rate,
                input=True,
                input_device_index=config.device_id,
                frames_per_buffer=config.chunk_size,
            )
            self._chunk = config.chunk_size
        except ImportError:
            raise ImportError("pip install pyaudio")

    def read_chunk(self) -> np.ndarray:
        raw = self._stream.read(self._chunk, exception_on_overflow=False)
        return np.frombuffer(raw, dtype=np.float32)

    def release(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()


class MockMicrophone(MicrophoneInterface):
    """
    Generates synthetic audio: silence, tones, and fake speech bursts.
    """

    def __init__(self, config: AudioConfig):
        self.sr    = config.sample_rate
        self.chunk = config.chunk_size
        self._t    = 0
        self._speech_phase = 0

    def read_chunk(self) -> np.ndarray:
        t = np.arange(self._t, self._t + self.chunk) / self.sr
        self._t += self.chunk

        # Simulate speech bursts every ~3 seconds
        speech_on = (self._t // (self.sr * 3)) % 2 == 0
        if speech_on:
            # Fake speech: sum of harmonics
            audio = (0.3 * np.sin(2 * np.pi * 200 * t) +
                     0.2 * np.sin(2 * np.pi * 400 * t) +
                     0.1 * np.sin(2 * np.pi * 800 * t) +
                     0.05 * np.random.randn(self.chunk))
        else:
            audio = 0.01 * np.random.randn(self.chunk)  # background noise

        return audio.astype(np.float32)
