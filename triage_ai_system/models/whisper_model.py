"""Whisper speech-to-text utilities for offline transcription."""

from __future__ import annotations

import tempfile
from pathlib import Path

import whisper


class WhisperTranscriber:
    """Loads a local Whisper model and transcribes audio files."""

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Transcribe raw audio bytes into text.

        Args:
            audio_bytes: Uploaded audio file as bytes.
            suffix: File extension for temporary file.

        Returns:
            Transcribed text string.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = Path(tmp.name)

        try:
            result = self.model.transcribe(str(temp_path), fp16=False)
            return result.get("text", "").strip()
        finally:
            if temp_path.exists():
                temp_path.unlink()
