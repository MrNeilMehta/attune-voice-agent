"""Tests for the dialogue manager and end-to-end agent flows."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from voice_agent import Appointment, NLUEngine, Patient, VoiceAgent  # noqa: E402
from voice_agent.asr import ASR  # noqa: E402
from voice_agent.dialogue import DialogueManager, State  # noqa: E402
from voice_agent.tts import TTS  # noqa: E402


def make_patient() -> Patient:
    return Patient(
        name="Maria Alvarez",
        appointment=Appointment(
            provider="Dr. Patel",
            specialty="cardiology",
            date="Thursday, June 18",
            time="10:30 AM",
            location="Riverside Clinic",
        ),
    )


class ListASR(ASR):
    def __init__(self, lines):
        self._it = iter(lines)

    def listen(self):
        return next(self._it, "goodbye")


class SilentTTS(TTS):
    def __init__(self):
        self.spoken = []

    def say(self, text):
        self.spoken.append(text)


nlu = NLUEngine()


def run_with(lines):
    agent = VoiceAgent(make_patient(), asr=ListASR(lines), tts=SilentTTS(), nlu=nlu)
    return agent.run()


def test_confirm_ends_call_without_handoff():
    t = run_with(["Yes that works"])
    assert not t.handoff
    assert any("confirmed" in s.lower() or "all set" in s.lower() for s in [x[1] for x in t.turns])


def test_emergency_triggers_handoff():
    t = run_with(["Actually I'm having chest pain"])
    assert t.handoff
    assert t.handoff_reason == "emergency"


def test_request_human_handoff():
    t = run_with(["Can I speak to a nurse"])
    assert t.handoff
    assert t.handoff_reason == "requested"


def test_reschedule_flow_captures_preference():
    t = run_with(["I need to reschedule", "Friday morning"])
    assert not t.handoff
    assert any("Friday morning" in turn_text for _, turn_text in t.turns)


def test_repeated_unknown_hands_off():
    t = run_with(["blah blah", "more nonsense"])
    assert t.handoff
    assert t.handoff_reason == "confused"


def test_question_then_confirm():
    t = run_with(["what time is it?", "great, confirm"])
    assert not t.handoff


def test_state_transitions_directly():
    dm = DialogueManager(make_patient())
    assert dm.state == State.START
    dm.start()
    assert dm.state == State.CONFIRMING
    dm.handle(nlu.understand("yes"))
    assert dm.state == State.ENDED
