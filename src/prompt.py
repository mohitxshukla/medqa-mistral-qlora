"""Single source of truth for Mistral-Instruct chat formatting.

Used by training, evaluation, the Gradio app, and the FastAPI server so the
prompt seen at inference matches the prompt seen during fine-tuning.
"""

from __future__ import annotations

# Medical assistant system prompt. Kept short — Mistral-Instruct v0.3 has no
# dedicated system role, so it is folded into the first [INST] block.
SYSTEM_PROMPT = (
    "You are a careful medical information assistant. Give clear, evidence-based "
    "answers in plain language. You do not provide diagnoses or prescriptions, and "
    "you remind the user to consult a licensed clinician for personal medical "
    "decisions."
)

DISCLAIMER = (
    "⚠️ Educational information only — not medical advice. "
    "Consult a licensed healthcare professional for any personal medical decision."
)


def build_prompt(question: str, system: str | None = SYSTEM_PROMPT) -> str:
    """Return the Mistral-Instruct prompt for a single user question.

    Format: ``<s>[INST] {system}\n\n{question} [/INST]`` — the answer is appended
    by the model at inference, or by the data layer (plus ``</s>``) at train time.
    """
    question = question.strip()
    if system:
        body = f"{system.strip()}\n\n{question}"
    else:
        body = question
    return f"<s>[INST] {body} [/INST]"


def build_training_text(question: str, answer: str, system: str | None = SYSTEM_PROMPT) -> str:
    """Full prompt + completion + EOS, used as the SFT training target."""
    return f"{build_prompt(question, system)} {answer.strip()}</s>"
