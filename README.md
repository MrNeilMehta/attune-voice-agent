# Healthcare Voice Agent (demo)

A small, modular voice agent that places a patient **appointment-confirmation
call**: it greets the patient, states their appointment, understands their
spoken reply, responds with warm and natural phrasing, and safely escalates to
a human when someone asks for one or signals an emergency.

It's a deliberately compact version of the kind of various comapnies build —
empathetic, compliant voice automation for healthcare — built to be readable,
testable, and easy to extend.

## What it does

```
Agent: Hi Maria, this is the care team with a quick reminder. You have a
       cardiology appointment with Dr. Patel on Thursday, June 18 at 10:30 AM,
       at Riverside Clinic. Can you let me know if that time still works?
You:   Actually I need to reschedule
Agent: That's completely fine. When would be a better time to come in?
You:   Friday morning
Agent: Thank you. I've noted that you'd prefer Friday morning. A scheduler will
       follow up shortly to lock in a new time that works.
```

## Architecture

The call is a pipeline of five swappable stages, each behind a small interface:

```
  ASR            NLU              Dialogue            NLG          TTS
(speech→text) (text→intent) (intent→decision) (decision→words) (text→speech)
   asr.py       nlu.py          dialogue.py        nlg.py        tts.py
        \           \               |               /            /
         \-----------\------- agent.py (orchestrates) ----------/
```

* **ASR** (`asr.py`) — `TextASR` (keyboard, default) or `WhisperASR` (mic +
  local faster-whisper).
* **NLU** (`nlu.py`) — a transparent rule-based intent classifier with negation
  handling (`"no I can't make it"` → cancel) and an emergency detector.
* **Dialogue** (`dialogue.py`) — a small state machine
  (`START → CONFIRMING → RESCHEDULING → ENDED/HANDOFF`) that makes every
  decision and **fails safe** to a human.
* **NLG** (`nlg.py`) — empathetic, lightly randomized response templates.
* **TTS** (`tts.py`) — `ConsoleTTS` (default) or `Pyttsx3TTS` (offline speech).

Because every stage is decoupled from audio, the same logic runs in the unit
tests and in the live voice loop — only the ASR/TTS ends change.

## Run it

No dependencies needed for the default text mode:

```bash
python run.py --demo      # auto-play a scripted conversation
python run.py             # type your own replies
```

Real microphone + spoken output (optional):

```bash
pip install -r requirements-voice.txt
python run.py --mode voice
```

Tests:

```bash
pip install -r requirements.txt
python -m pytest
```

## Design choices worth knowing

* **Rule-based NLU on purpose.** For a demo it's zero-dependency,
  deterministic, and easy to test. The `NLUEngine.understand()` interface is
  the seam: drop in an embedding- or LLM-based classifier later without
  touching the dialogue layer.
* **Safety first.** Emergencies, explicit requests for a person, and repeated
  confusion all route to a human. The agent never gives medical advice — on a
  distress cue it directs the caller to emergency services and hands off.
* **Separation of concerns.** "What to do" (dialogue) is independent of "how to
  say it" (NLG) and "how to hear/speak it" (ASR/TTS), so tone, language, and
  audio backends can all change independently.

## Where I'd take it next

* Real date/time parsing in the reschedule step (e.g. `dateparser`) and a
  calendar lookup for open slots.
* A learned NLU model trained on labeled patient utterances, measured against
  the rule-based baseline.
* Barge-in / streaming ASR so the patient can interrupt.
* Structured call logging + an evaluation harness over recorded transcripts.

## Layout

```
attune-voice-agent/
├── run.py                     # CLI entry point
├── data/sample_patient.json   # example patient + appointment
├── src/voice_agent/
│   ├── models.py              # dataclasses + Intent enum
│   ├── asr.py  nlu.py  dialogue.py  nlg.py  tts.py
│   └── agent.py               # orchestrator + transcript
└── tests/                     # pytest suite (NLU + dialogue + flows)
```
