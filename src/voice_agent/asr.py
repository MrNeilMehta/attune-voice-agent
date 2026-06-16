"""Speech-to-text (ASR) backends behind one small interface.

`TextASR` is the default and needs no dependencies: the "microphone" is the
keyboard. `WhisperASR` records from a real mic and transcribes locally with
faster-whisper. It is imported lazily so the project runs out of the box and
only pulls in heavy audio/ML packages when you actually choose voice mode.

To add a cloud STT provider, implement `listen()` in a new subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ASR(ABC):
    @abstractmethod
    def listen(self) -> str:
        """Capture one patient turn and return it as text."""


class TextASR(ASR):
    """Reads a turn from stdin. Ideal for development and tests."""

    def __init__(self, prompt: str = "You: ") -> None:
        self.prompt = prompt

    def listen(self) -> str:
        try:
            return input(self.prompt)
        except EOFError:
            return "goodbye"


class WhisperASR(ASR):
    """Records from the microphone and transcribes with faster-whisper.

    Requires the optional voice extras (see requirements-voice.txt):
        pip install faster-whisper sounddevice numpy
    """

    def __init__(
        self,
        model_size: str = "base.en",
        seconds: float = 5.0,
        samplerate: int = 16000,
    ) -> None:
        try:
            import numpy  # noqa: F401
            import sounddevice  # noqa: F401
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "Voice mode needs extra packages. Install them with:\n"
                "    pip install faster-whisper sounddevice numpy"
            ) from exc

        self._sd = sounddevice
        self._np = numpy
        self.seconds = seconds
        self.samplerate = samplerate
        self.model = WhisperModel(model_size, compute_type="int8")

    def listen(self) -> str:  # pragma: no cover - requires a microphone
        print(f"[listening for {self.seconds:.0f}s...]")
        audio = self._sd.rec(
            int(self.seconds * self.samplerate),
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
        )
        self._sd.wait()
        samples = self._np.squeeze(audio)
        segments, _ = self.model.transcribe(samples, language="en")
        return " ".join(segment.text for segment in segments).strip()
