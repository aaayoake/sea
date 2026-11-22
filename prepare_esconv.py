import json
import random

SEEKER_OPENING_PROMPT = (
    "Begin a new emotional-support conversation with a natural opening in your own words. "
    "Do not quote these instructions or reveal all of your backstory at once."
)


def load_local_esconv(json_path: str) -> list[dict]:
    """Load ESConv JSONL exported by Hugging Face Datasets."""
    dialogues = []
    with open(json_path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            data = json.loads(row["text"]) if isinstance(row.get("text"), str) else row
            dialogues.append(data)
    return dialogues


def split_dialogues(
    dialogues: list[dict], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    if len(dialogues) < 2:
        raise ValueError("At least two dialogues are required to create train and validation splits")

    shuffled = list(dialogues)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    split_idx = min(max(split_idx, 1), len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def merge_consecutive_turns(dialogue: dict) -> list[dict]:
    merged = []
    for turn in dialogue.get("dialog", []):
        speaker = turn["speaker"]
        text = turn["text"].strip()
        if not text:
            continue
        if merged and merged[-1]["speaker"] == speaker:
            merged[-1]["content"] += " " + text
        else:
            merged.append({"speaker": speaker, "content": text})
    return merged


def build_system_prompt(emotion: str, problem: str, situation: str) -> str:
    return (
        "You are simulating a real human seeker of emotional support. "
        "Respond naturally as the person described; do not act as a counselor.\n"
        f"Problem Category: {problem}\n"
        f"Primary Emotion: {emotion}\n"
        f"Backstory (reveal gradually, do NOT dump all at once): {situation}\n"
        "Only share specific backstory details when directly prompted or relevant. "
        "Keep responses conversational, usually 1-2 short sentences. "
        "Express the situation freshly; do not copy stock example dialogue."
    )


def build_supporter_system_prompt() -> str:
    return (
        "You are a compassionate peer supporter in an online emotional-support chat. "
        "Listen carefully, validate the seeker's feelings, ask natural follow-up questions, "
        "and offer gentle support only when appropriate. Keep responses conversational and concise."
    )
