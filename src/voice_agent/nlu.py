"""Natural-language understanding: turn a raw utterance into an Intent.

This uses a transparent, rule-based classifier rather than a trained model.
That is a deliberate choice for a demo: it has zero dependencies, is fully
deterministic, and is trivial to unit-test and explain. The `NLUEngine`
interface is intentionally small so a model-based classifier (embeddings or
an LLM) can be dropped in later without touching the rest of the pipeline.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .models import Intent, NLUResult

# Keyword sets per intent. Ordering in `_INTENT_RULES` encodes priority:
# earlier intents win ties, which lets us put more specific intents first.
_INTENT_KEYWORDS: Dict[Intent, List[str]] = {
    Intent.REQUEST_HUMAN: [
        "human", "person", "someone", "representative", "agent",
        "nurse", "doctor", "staff", "real person", "operator",
    ],
    Intent.RESCHEDULE: [
        "reschedule", "move", "change", "different day", "different time",
        "another day", "another time", "push back", "earlier", "later",
    ],
    Intent.CANCEL: [
        "cancel", "can't make", "cannot make", "won't make", "will not make",
        "can't come", "cannot come", "skip", "not coming", "not able to come",
    ],
    Intent.ASK_TIME: [
        "what time", "when is", "when's", "remind me when", "which day",
        "what day", "how late", "how early",
    ],
    Intent.ASK_LOCATION: [
        "where", "what address", "which clinic", "location", "directions",
        "how do i get", "what building",
    ],
    Intent.CONFIRM: [
        "confirm", "yes", "yep", "yeah", "sure", "i'll be there",
        "ill be there", "i will be there", "see you then", "sounds good",
        "that works", "correct", "affirmative", "keep it", "still good",
    ],
    Intent.GOODBYE: [
        "bye", "goodbye", "that's all", "thats all", "nothing else",
        "we're done", "were done", "thank you bye",
    ],
}

# Phrases that flip an otherwise-affirmative reply into a cancel.
_NEGATIONS = ["no", "not", "can't", "cannot", "won't", "will not", "don't", "do not"]

# Cues that should immediately route to a human / emergency services.
# This is a safety net, not medical triage: the agent never advises, it escalates.
_EMERGENCY_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "trouble breathing",
    "suicidal", "kill myself", "hurt myself", "overdose", "overdosed",
    "bleeding", "unconscious", "passed out", "stroke", "heart attack",
    "911", "emergency", "severe pain", "can't move",
]


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation so keyword matching is robust."""
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9\s']", " ", text)


def detect_emergency(text: str) -> bool:
    norm = _normalize(text)
    return any(kw in norm for kw in _EMERGENCY_KEYWORDS)


def _score_intent(norm: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """Return a match score and the keywords that matched."""
    matched = [kw for kw in keywords if kw in norm]
    if not matched:
        return 0.0, []
    # Confidence grows with the strength of the longest matched phrase.
    longest = max(len(kw.split()) for kw in matched)
    score = min(1.0, 0.5 + 0.25 * longest + 0.1 * (len(matched) - 1))
    return score, matched


class NLUEngine:
    """Maps an utterance to an :class:`NLUResult`.

    Swap this class out (keeping the ``understand`` signature) to plug in an
    embedding- or LLM-based classifier without changing the dialogue layer.
    """

    def understand(self, text: str) -> NLUResult:
        if not text or not text.strip():
            return NLUResult(Intent.UNKNOWN, 0.0)

        if detect_emergency(text):
            return NLUResult(
                Intent.REQUEST_HUMAN, 1.0, matched=["<emergency>"], is_emergency=True
            )

        norm = _normalize(text)
        has_negation = any(re.search(rf"\b{re.escape(n)}\b", norm) for n in _NEGATIONS)

        best_intent = Intent.UNKNOWN
        best_score = 0.0
        best_matched: List[str] = []

        for intent, keywords in _INTENT_KEYWORDS.items():
            score, matched = _score_intent(norm, keywords)
            if score > best_score:
                best_intent, best_score, best_matched = intent, score, matched

        # "No, I can't make it" should read as CANCEL even though "make" etc.
        # may look affirmative. If we landed on CONFIRM but there's a negation,
        # reinterpret as CANCEL.
        if best_intent == Intent.CONFIRM and has_negation:
            best_intent = Intent.CANCEL
            best_matched = best_matched + ["<negation>"]

        # A bare "no" with nothing else is a cancel of the confirmation.
        if best_intent == Intent.UNKNOWN and has_negation:
            best_intent, best_score, best_matched = Intent.CANCEL, 0.6, ["no"]

        return NLUResult(
            best_intent,
            round(best_score, 2),
            matched=best_matched,
            entities={"raw_text": text.strip()},
        )
