"""
ChimeraRobot Speech I/O
------------------------
Fully offline speech input and output — no cloud APIs.

Text-to-Speech (TTS):
  • pyttsx3  — cross-platform, works immediately, robotic but functional
  • espeak   — lightweight, very fast, good for embedded systems
  • Emotion-aware: voice pitch/speed changes with emotional state
    (happy → faster + higher pitch, sad → slower + lower pitch)

Speech-to-Text (STT):
  • Vosk     — offline neural STT, decent accuracy, no internet
  • Whisper  — high accuracy offline STT (OpenAI Whisper, runs locally)
  • Falls back to silence if no engine available

Voice personality:
  The robot's voice reflects its emotional state — a key part of
  making it feel alive. Fear makes it speak faster and quieter.
  Joy makes it speak louder and higher-pitched.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from robot_config import SpeechConfig


# ─────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SpeechResult:
    text:        str
    confidence:  float        # 0–1
    is_final:    bool         # Is this a final or partial result?
    timestamp:   float = 0.0


@dataclass
class VoiceParams:
    """Parameters that control how the robot speaks."""
    rate:   int   = 150    # Words per minute
    pitch:  int   = 50     # 0–100 (engine-specific)
    volume: float = 1.0    # 0–1


# ─────────────────────────────────────────────────────────────────────
#  Emotion → voice mapping
# ─────────────────────────────────────────────────────────────────────

EMOTION_VOICE: dict = {
    "JOY":         VoiceParams(rate=175, pitch=65,  volume=1.0),
    "TRUST":       VoiceParams(rate=145, pitch=50,  volume=0.9),
    "FEAR":        VoiceParams(rate=195, pitch=70,  volume=0.7),
    "SURPRISE":    VoiceParams(rate=190, pitch=75,  volume=1.0),
    "SADNESS":     VoiceParams(rate=120, pitch=35,  volume=0.75),
    "DISGUST":     VoiceParams(rate=140, pitch=40,  volume=0.85),
    "ANGER":       VoiceParams(rate=165, pitch=30,  volume=1.0),
    "ANTICIPATION":VoiceParams(rate=160, pitch=55,  volume=0.95),
    "NEUTRAL":     VoiceParams(rate=150, pitch=50,  volume=0.9),
}


# ─────────────────────────────────────────────────────────────────────
#  Text-to-speech engines
# ─────────────────────────────────────────────────────────────────────

class BaseTTS:
    def speak(self, text: str, params: VoiceParams): ...
    def speak_async(self, text: str, params: VoiceParams): ...
    def stop(self): ...
    def is_speaking(self) -> bool: ...


class Pyttsx3TTS(BaseTTS):
    """
    pyttsx3 — cross-platform TTS using OS native engines.
    Windows: SAPI5 | Linux: eSpeak | macOS: NSSpeechSynthesizer
    Install: pip install pyttsx3
    """

    def __init__(self, config: SpeechConfig):
        import pyttsx3
        self._engine  = pyttsx3.init()
        self._lock    = threading.Lock()
        self._speaking = False

        # Set voice by language if possible
        voices = self._engine.getProperty("voices")
        for v in (voices or []):
            if config.language in (v.languages[0] if v.languages else ""):
                self._engine.setProperty("voice", v.id)
                break

    def speak(self, text: str, params: VoiceParams):
        """Blocking speech."""
        with self._lock:
            self._speaking = True
            self._engine.setProperty("rate",   params.rate)
            self._engine.setProperty("pitch",  params.pitch)
            self._engine.setProperty("volume", params.volume)
            self._engine.say(text)
            self._engine.runAndWait()
            self._speaking = False

    def speak_async(self, text: str, params: VoiceParams):
        """Non-blocking: speaks in a background thread."""
        t = threading.Thread(target=self.speak, args=(text, params), daemon=True)
        t.start()

    def stop(self):
        self._engine.stop()
        self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking


class EspeakTTS(BaseTTS):
    """
    eSpeak NG — very lightweight, fast, works on Raspberry Pi.
    Install: sudo apt install espeak-ng  (Linux)
             brew install espeak        (macOS)
    """

    def __init__(self, config: SpeechConfig):
        import subprocess
        self._lang    = config.language
        self._proc:   Optional[subprocess.Popen] = None
        self._speaking = False
        # Test availability
        try:
            subprocess.run(["espeak", "--version"], capture_output=True, timeout=2)
            self._cmd = "espeak"
        except Exception:
            try:
                subprocess.run(["espeak-ng", "--version"], capture_output=True, timeout=2)
                self._cmd = "espeak-ng"
            except Exception:
                raise RuntimeError("espeak / espeak-ng not found. Install it or use pyttsx3.")

    def _build_cmd(self, text: str, params: VoiceParams) -> list:
        return [
            self._cmd,
            "-l", self._lang,
            "-s", str(params.rate),
            "-p", str(params.pitch),
            "-a", str(int(params.volume * 200)),
            text,
        ]

    def speak(self, text: str, params: VoiceParams):
        import subprocess
        self._speaking = True
        self._proc = subprocess.Popen(self._build_cmd(text, params))
        self._proc.wait()
        self._speaking = False

    def speak_async(self, text: str, params: VoiceParams):
        t = threading.Thread(target=self.speak, args=(text, params), daemon=True)
        t.start()

    def stop(self):
        if self._proc:
            self._proc.terminate()
        self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking


class SilentTTS(BaseTTS):
    """Fallback: prints text instead of speaking. Always available."""

    def speak(self, text: str, params: VoiceParams):
        print(f"[Robot speaks]: {text}")

    def speak_async(self, text: str, params: VoiceParams):
        self.speak(text, params)

    def stop(self): pass

    def is_speaking(self) -> bool:
        return False


# ─────────────────────────────────────────────────────────────────────
#  Speech-to-text engines
# ─────────────────────────────────────────────────────────────────────

class BaseSTT:
    def start_listening(self, callback: Callable[[SpeechResult], None]): ...
    def stop_listening(self): ...
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> SpeechResult: ...


class VoskSTT(BaseSTT):
    """
    Vosk — offline neural STT. Fast, accurate, 50MB model.
    Works on Raspberry Pi, no internet needed.
    Install: pip install vosk
    Model download: https://alphacephei.com/vosk/models
    """

    def __init__(self, model_path: str = "models/vosk-model-en"):
        try:
            from vosk import Model, KaldiRecognizer
            import json as _json
            self._Model    = Model
            self._KaldiRec = KaldiRecognizer
            self._json     = _json

            if not __import__("os").path.exists(model_path):
                raise FileNotFoundError(
                    f"Vosk model not found at '{model_path}'.\n"
                    "Download from: https://alphacephei.com/vosk/models\n"
                    "Recommended: vosk-model-small-en-us"
                )
            self._model = Model(model_path)
            self._recogniser = KaldiRecognizer(self._model, 16000)
            self._listening  = False
            self._callback: Optional[Callable] = None
            print("[STT] Vosk loaded successfully")
        except ImportError:
            raise ImportError("pip install vosk")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> SpeechResult:
        """Transcribe a numpy audio array."""
        # Convert float32 → int16 PCM
        pcm = (audio * 32767).astype(np.int16).tobytes()
        if self._recogniser.AcceptWaveform(pcm):
            result = self._json.loads(self._recogniser.Result())
            text   = result.get("text", "")
            conf   = result.get("confidence", 0.8) if text else 0.0
        else:
            partial = self._json.loads(self._recogniser.PartialResult())
            text    = partial.get("partial", "")
            conf    = 0.5 if text else 0.0

        return SpeechResult(
            text=text.strip(),
            confidence=conf,
            is_final=True,
            timestamp=time.time(),
        )

    def start_listening(self, callback: Callable[[SpeechResult], None]):
        self._listening = True
        self._callback  = callback

    def stop_listening(self):
        self._listening = False


class WhisperSTT(BaseSTT):
    """
    OpenAI Whisper — higher accuracy, runs fully offline.
    Install: pip install openai-whisper
    Sizes: tiny(~75MB), base(~142MB), small(~466MB), medium(~1.5GB)
    """

    def __init__(self, model_size: str = "base"):
        try:
            import whisper
            print(f"[STT] Loading Whisper {model_size}...")
            self._model = whisper.load_model(model_size)
            print("[STT] Whisper ready")
        except ImportError:
            raise ImportError("pip install openai-whisper")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> SpeechResult:
        result = self._model.transcribe(audio.astype(np.float32))
        text   = result.get("text", "").strip()
        return SpeechResult(
            text=text,
            confidence=0.9 if text else 0.0,
            is_final=True,
            timestamp=time.time(),
        )

    def start_listening(self, callback): pass
    def stop_listening(self):           pass


class MockSTT(BaseSTT):
    """Returns pre-scripted responses for simulation mode."""

    _SCRIPTS = [
        "Hello robot, what do you see?",
        "Can you pick up the cup?",
        "How are you feeling?",
        "Walk forward please.",
        "That is interesting.",
        "Good job!",
    ]

    def __init__(self):
        self._idx = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> SpeechResult:
        text = self._SCRIPTS[self._idx % len(self._SCRIPTS)]
        self._idx += 1
        return SpeechResult(text=text, confidence=1.0, is_final=True, timestamp=time.time())

    def start_listening(self, callback): pass
    def stop_listening(self):           pass


# ─────────────────────────────────────────────────────────────────────
#  Speech system facade
# ─────────────────────────────────────────────────────────────────────

class SpeechSystem:
    """
    Unified speech I/O system.
    Handles emotion-aware TTS and STT with automatic engine fallback.
    """

    def __init__(self, config: SpeechConfig, simulation: bool = True):
        self.config = config

        # Init TTS
        if simulation:
            self._tts = SilentTTS()
        else:
            self._tts = self._init_tts(config)

        # Init STT
        if simulation:
            self._stt = MockSTT()
        else:
            self._stt = self._init_stt(config)

        self._speech_queue: queue.Queue = queue.Queue()
        self._listening     = False
        self._history: List[SpeechResult] = []

    @staticmethod
    def _init_tts(config: SpeechConfig) -> BaseTTS:
        for cls, name in [(EspeakTTS, "eSpeak"), (Pyttsx3TTS, "pyttsx3")]:
            try:
                engine = cls(config)
                print(f"[TTS] Using {name}")
                return engine
            except Exception as e:
                print(f"[TTS] {name} unavailable: {e}")
        print("[TTS] Falling back to silent mode")
        return SilentTTS()

    @staticmethod
    def _init_stt(config: SpeechConfig) -> BaseSTT:
        for cls, name, kwargs in [
            (VoskSTT,   "Vosk",    {"model_path": "models/vosk-model-en"}),
            (WhisperSTT,"Whisper", {"model_size": "base"}),
        ]:
            try:
                engine = cls(**kwargs)
                print(f"[STT] Using {name}")
                return engine
            except Exception as e:
                print(f"[STT] {name} unavailable: {e}")
        print("[STT] Falling back to mock mode")
        return MockSTT()

    def speak(self, text: str, emotion: str = "NEUTRAL", blocking: bool = False):
        """
        Speak text with emotion-appropriate voice.
        emotion: name from Emotion enum (e.g. "JOY", "FEAR")
        """
        params = EMOTION_VOICE.get(emotion.upper(), EMOTION_VOICE["NEUTRAL"])
        if blocking:
            self._tts.speak(text, params)
        else:
            self._tts.speak_async(text, params)

    def listen(self, audio: np.ndarray, sample_rate: int = 16000) -> SpeechResult:
        """Transcribe an audio chunk."""
        result = self._stt.transcribe(audio, sample_rate)
        if result.text:
            self._history.append(result)
            print(f"[STT] Heard: '{result.text}' (conf={result.confidence:.2f})")
        return result

    def stop_speaking(self):
        self._tts.stop()

    def is_speaking(self) -> bool:
        return self._tts.is_speaking()

    def last_heard(self) -> Optional[str]:
        return self._history[-1].text if self._history else None

    def history(self, n: int = 10) -> List[str]:
        return [r.text for r in self._history[-n:]]


# ─────────────────────────────────────────────────────────────────────
#  Word-by-word streaming TTS
# ─────────────────────────────────────────────────────────────────────

class StreamingTTS:
    """
    Speaks text word-by-word as it's generated, so the robot starts
    speaking before the full response is ready (reduces latency).
    """

    def __init__(self, tts: BaseTTS):
        self._tts     = tts
        self._buffer  = []
        self._thread: Optional[threading.Thread] = None
        self._done    = False

    def feed(self, text_fragment: str, params: VoiceParams):
        """Feed a new text fragment (e.g. from streaming LLM)."""
        self._buffer.append(text_fragment)
        # Start speaking when we have a full sentence
        combined = "".join(self._buffer)
        if any(combined.endswith(p) for p in [".", "!", "?", ","]):
            to_say = combined.strip()
            self._buffer.clear()
            if not self._tts.is_speaking():
                self._tts.speak_async(to_say, params)

    def flush(self, params: VoiceParams):
        """Speak any remaining buffered text."""
        if self._buffer:
            self._tts.speak("".join(self._buffer), params)
            self._buffer.clear()
