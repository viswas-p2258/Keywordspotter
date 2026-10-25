"""Local English speech-to-text transcription for uploaded WAV files."""

from functools import lru_cache

import av
from faster_whisper import WhisperModel


class TranscriptionError(Exception):
    """Raised when an audio file cannot be transcribed."""


@lru_cache(maxsize=1)
def _load_model() -> WhisperModel:
    try:
        return WhisperModel("base.en", device="cpu", compute_type="int8")
    except (OSError, RuntimeError, ValueError) as exc:
        raise TranscriptionError(
            "Could not load the local English speech model. Check your internet "
            "connection for the first download, then try again."
        ) from exc


def transcribe_audio(audio_path: str) -> str:
    """Transcribe an English audio file and return its recognized words."""
    try:
        segments, _ = _load_model().transcribe(
            audio_path,
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
    except (av.error.FFmpegError, OSError, RuntimeError, ValueError) as exc:
        raise TranscriptionError(
            "Could not read or transcribe this WAV file. Check that it contains "
            "clear English speech and try another recording."
        ) from exc
