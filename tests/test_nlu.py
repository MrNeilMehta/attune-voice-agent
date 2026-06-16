"""Tests for the rule-based NLU engine."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from voice_agent.models import Intent  # noqa: E402
from voice_agent.nlu import NLUEngine, detect_emergency  # noqa: E402

nlu = NLUEngine()


def test_confirm_variants():
    for text in ["Yes", "yep that works", "I'll be there", "sounds good"]:
        assert nlu.understand(text).intent == Intent.CONFIRM


def test_cancel_variants():
    for text in ["I need to cancel", "I can't make it", "no I won't make it"]:
        assert nlu.understand(text).intent == Intent.CANCEL


def test_bare_no_is_cancel():
    assert nlu.understand("no").intent == Intent.CANCEL


def test_reschedule():
    assert nlu.understand("can we move it to another day").intent == Intent.RESCHEDULE


def test_questions():
    assert nlu.understand("what time is it again?").intent == Intent.ASK_TIME
    assert nlu.understand("where is the clinic?").intent == Intent.ASK_LOCATION


def test_request_human():
    assert nlu.understand("can I talk to a real person").intent == Intent.REQUEST_HUMAN


def test_unknown():
    assert nlu.understand("the weather is nice today").intent == Intent.UNKNOWN


def test_emergency_detection():
    assert detect_emergency("I'm having chest pain")
    result = nlu.understand("I think I'm having a heart attack")
    assert result.is_emergency
    assert result.intent == Intent.REQUEST_HUMAN


def test_raw_text_preserved():
    assert nlu.understand("Friday morning").entities["raw_text"] == "Friday morning"
