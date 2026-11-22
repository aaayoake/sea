import json
from pathlib import Path

from prepare_esconv import (
    SEEKER_OPENING_PROMPT,
    build_system_prompt,
    load_local_esconv,
    merge_consecutive_turns,
    split_dialogues,
)


def process_esconv(
    input_path: str = "esconv_train.json",
    output_dir: str = "data",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_dialogues, val_dialogues = split_dialogues(
        load_local_esconv(input_path), val_ratio, seed
    )

    for split_name, dialogues in (("train", train_dialogues), ("val", val_dialogues)):
        out_path = output_path / f"esconv_sft_{split_name}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as file:
            for dialogue in dialogues:
                emotion = dialogue.get("emotion_type", "distressed")
                problem = dialogue.get("problem_type", "personal issue")
                situation = dialogue.get("situation", "").strip()
                history = []
                has_seeker_turn = False

                for turn in merge_consecutive_turns(dialogue):
                    if turn["speaker"] == "sys":
                        if has_seeker_turn:
                            history.append({"role": "user", "content": turn["content"]})
                    elif turn["speaker"] == "usr":
                        if not has_seeker_turn:
                            history = [{"role": "user", "content": SEEKER_OPENING_PROMPT}]

                        messages = [
                            {
                                "role": "system",
                                "content": build_system_prompt(
                                    emotion, problem, situation
                                ),
                            },
                            *history,
                            {"role": "assistant", "content": turn["content"]},
                        ]
                        record = {
                            "messages": messages,
                            "emotion": emotion,
                            "problem": problem,
                        }
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        count += 1
                        history.append({"role": "assistant", "content": turn["content"]})
                        has_seeker_turn = True

        print(f"Wrote {count} seeker SFT examples to {out_path}")


if __name__ == "__main__":
    process_esconv()
