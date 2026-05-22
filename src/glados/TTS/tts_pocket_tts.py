"""
PocketTTS.cpp integration via subprocess.

Wraps the PocketTTS.cpp CLI executable to generate speech audio from text.
Uses the --stdout flag to pipe raw f32le PCM directly into numpy.
"""

from dataclasses import dataclass
import shutil
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..utils.resources import resource_path


@dataclass
class PocketTTSConfig:
    """Configuration for PocketTTS.cpp integration."""

    executable: Path
    """Path to the pocket-tts executable."""

    models_dir: Path
    """Path to the ONNX models directory."""

    voices_dir: Path
    """Path to the voices directory (contains voice samples and cache)."""

    tokenizer: Path
    """Path to the SentencePiece tokenizer model."""

    precision: str = "int8"
    """Model precision: 'int8' or 'fp32'."""

    temperature: float = 0.7
    """Sampling temperature."""

    lsd_steps: int = 1
    """Flow matching ODE solver steps."""

    threads: int = 0
    """Thread budget (0 = half of available cores)."""

    sample_rate: int = 24000
    """Output audio sample rate (PocketTTS.cpp default)."""


# Default paths
_DEFAULT_EXECUTABLE = resource_path("deps/PocketTTS.cpp/pocket-tts")
_DEFAULT_MODELS_DIR = resource_path("models/TTS")
_DEFAULT_VOICES_DIR = resource_path("data/voices")
_DEFAULT_TOKENIZER = resource_path("models/TTS/tokenizer.model")


def _find_executable() -> Path:
    """Find the PocketTTS.cpp executable."""
    if _DEFAULT_EXECUTABLE.exists():
        return _DEFAULT_EXECUTABLE
    # Fallback: search in PATH
    found = shutil.which("pocket-tts")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "PocketTTS.cpp executable not found. "
        "Expected at: " + str(_DEFAULT_EXECUTABLE) + "\n"
        "Build it with: scripts/build_pocket_tts.sh"
    )


def _ensure_voices_dir(voices_dir: Path) -> None:
    """Ensure the voices directory exists."""
    voices_dir.mkdir(parents=True, exist_ok=True)


def _find_default_voice(voices_dir: Path) -> Path:
    """Find a default voice sample in the voices directory.

    Looks for .wav, .mp3, or .flac files. Returns the first one found.
    Raises FileNotFoundError if no voice sample is available.
    """
    for ext in ("*.wav", "*.mp3", "*.flac"):
        for f in voices_dir.glob(ext):
            if f.is_file():
                return f
    raise FileNotFoundError(
        f"No voice sample found in {voices_dir}. "
        "Place a .wav, .mp3, or .flac file in the voices directory. "
        "You can use any short audio recording of the desired voice."
    )


class SpeechSynthesizer:
    """
    PocketTTS.cpp-based speech synthesizer for text-to-speech conversion.

    This class wraps the PocketTTS.cpp CLI executable via subprocess to generate
    speech audio from text. It supports voice cloning from short audio samples
    and uses the --stdout flag for efficient raw PCM piping.

    Attributes:
        SAMPLE_RATE: Audio sample rate (24000 Hz, PocketTTS.cpp default)
    """

    SAMPLE_RATE: int = 24000

    def __init__(
        self,
        voice: str | Path | None = None,
        executable: Path | None = None,
        models_dir: Path | None = None,
        voices_dir: Path | None = None,
        tokenizer: Path | None = None,
        precision: str = "int8",
        temperature: float = 0.7,
        lsd_steps: int = 1,
        threads: int = 0,
    ) -> None:
        """
        Initialize the PocketTTS synthesizer.

        Parameters:
            voice: Voice sample path or name. If a name, looks in voices_dir.
                   If None, finds the first voice file in voices_dir.
            executable: Path to pocket-tts executable. Defaults to project path.
            models_dir: Path to ONNX models directory.
            voices_dir: Path to voices directory.
            tokenizer: Path to tokenizer.model file.
            precision: Model precision ('int8' or 'fp32').
            temperature: Sampling temperature (0.0 to 1.0).
            lsd_steps: Flow matching ODE solver steps.
            threads: Thread budget (0 = auto).
        """
        self.sample_rate = self.SAMPLE_RATE
        self.precision = precision
        self.temperature = temperature
        self.lsd_steps = lsd_steps
        self.threads = threads

        # Resolve paths
        self.executable = executable or _find_executable()
        self.models_dir = models_dir or _DEFAULT_MODELS_DIR
        self.voices_dir = voices_dir or _DEFAULT_VOICES_DIR
        self.tokenizer = tokenizer or _DEFAULT_TOKENIZER

        # Resolve voice
        _ensure_voices_dir(self.voices_dir)
        if voice is None:
            self.voice_path = _find_default_voice(self.voices_dir)
        elif isinstance(voice, Path):
            self.voice_path = voice
        else:
            # Try as filename in voices_dir first, then as absolute path
            candidate = self.voices_dir / voice
            if candidate.exists():
                self.voice_path = candidate
            else:
                self.voice_path = Path(voice)

        if not self.voice_path.exists():
            raise FileNotFoundError(
                f"Voice sample not found: {self.voice_path}. "
                f"Looked in voices_dir: {self.voices_dir}"
            )

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        """
        Convert input text to synthesized speech audio.

        Runs the PocketTTS.cpp executable with --stdout to pipe raw f32le PCM
        directly into a numpy array.

        Parameters:
            text (str): The text to be converted to speech.

        Returns:
            NDArray[np.float32]: An array of audio samples representing the synthesized speech.
        """
        # Build the command
        cmd = [
            str(self.executable),
            "--precision", self.precision,
            "--temperature", str(self.temperature),
            "--lsd-steps", str(self.lsd_steps),
            "--models-dir", str(self.models_dir),
            "--tokenizer", str(self.tokenizer),
        ]

        if self.threads > 0:
            cmd.extend(["--threads", str(self.threads)])

        cmd.extend(["--stdout", text, str(self.voice_path)])

        # Run the subprocess and capture raw f32le PCM from stdout
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,  # Generous timeout for TTS generation
        )

        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"PocketTTS.cpp failed (exit code {result.returncode}): {stderr_text}"
            )

        # Parse raw f32le PCM from stdout
        pcm_data = result.stdout
        if len(pcm_data) == 0:
            return np.array([], dtype=np.float32)

        # f32le = 4 bytes per sample, little-endian
        audio = np.frombuffer(pcm_data, dtype=np.float32)
        return audio

    def __repr__(self) -> str:
        return (
            f"SpeechSynthesizer(engine='pocket-tts', "
            f"voice='{self.voice_path.name}', "
            f"precision='{self.precision}', "
            f"sample_rate={self.sample_rate})"
        )
