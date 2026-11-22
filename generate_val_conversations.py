import argparse
import json
import random
import re
from pathlib import Path

from prepare_esconv import load_local_esconv
from run_live_rollouts import generate_reply, simulate_conversation

EMOTION_LABELS = (
    "anger",
    "anxiety",
    "depression",
    "disgust",
    "fear",
    "nervousness",
    "sadness",
    "shame",
)


def select_validation_scenarios(
    input_file: str, num_conversations: int, seed: int
) -> list[dict]:
    dialogues = load_local_esconv(input_file)
    if num_conversations < 1:
        raise ValueError("num-conversations must be positive")
    if num_conversations > len(dialogues):
        raise ValueError(
            f"Requested {num_conversations} conversations but the validation file "
            f"contains only {len(dialogues)} records"
        )

    scenarios = []
    for source_index, dialogue in enumerate(dialogues):
        problem = dialogue.get("problem_type")
        situation = dialogue.get("situation")
        reference_emotion = dialogue.get("emotion_type")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (problem, situation, reference_emotion)
        ):
            raise ValueError(
                f"Validation record {source_index} must have non-empty "
                "problem_type, situation, and emotion_type fields"
            )
        scenarios.append({
            "source_index": source_index,
            "problem": problem.strip(),
            "situation": situation.strip(),
            "reference_emotion": reference_emotion.strip().lower(),
        })

    return random.Random(seed).sample(scenarios, num_conversations)


def build_emotion_inference_messages(problem: str, situation: str) -> list[dict]:
    labels = ", ".join(EMOTION_LABELS)
    return [
        {
            "role": "system",
            "content": (
                "You are identifying the primary emotion a person may be experiencing "
                "from a brief support-seeking scenario. Choose the single best-fitting "
                f"label from this list: {labels}. Return exactly the label, with no "
                "explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Problem category: {problem}\nSituation: {situation}",
        },
    ]


def parse_inferred_emotion(response: str) -> str:
    normalized_response = response.strip().lower()
    matches = {
        label
        for label in EMOTION_LABELS
        if re.search(rf"\b{re.escape(label)}\b", normalized_response)
    }
    if len(matches) != 1:
        raise ValueError(
            "Could not identify exactly one allowed emotion label in model response: "
            f"{response!r}"
        )
    return matches.pop()


def infer_emotion(model, tokenizer, problem: str, situation: str) -> str:
    model.set_adapter("seeker")
    response = generate_reply(
        model,
        tokenizer,
        build_emotion_inference_messages(problem, situation),
        max_new_tokens=16,
    )
    return parse_inferred_emotion(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate complete seeker/supporter conversations from ESConv validation "
            "problems and situations, inferring the seeker's emotion without showing "
            "the reference emotion or transcript to the model."
        )
    )
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--seeker-adapter", required=True)
    parser.add_argument("--supporter-adapter", required=True)
    parser.add_argument("--input-file", default="esconv_val.json")
    parser.add_argument("--output-file", default="eval_outputs/val_live_rollouts.jsonl")
    parser.add_argument("--num-conversations", type=int, default=50)
    parser.add_argument("--min-turns", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.min_turns < 1 or args.max_turns < args.min_turns:
        raise ValueError("Turn counts must satisfy 1 <= min-turns <= max-turns")
    scenarios = select_validation_scenarios(
        args.input_file, args.num_conversations, args.seed
    )
    if not Path(args.seeker_adapter).is_dir():
        raise FileNotFoundError(f"Seeker adapter directory does not exist: {args.seeker_adapter}")
    if not Path(args.supporter_adapter).is_dir():
        raise FileNotFoundError(
            f"Supporter adapter directory does not exist: {args.supporter_adapter}"
        )

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading cached base model {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )

    print("Attaching seeker and supporter adapters...")
    model = PeftModel.from_pretrained(
        base_model, args.seeker_adapter, adapter_name="seeker", local_files_only=True
    )
    model.load_adapter(
        args.supporter_adapter, adapter_name="supporter", local_files_only=True
    )
    model.eval()

    with open(args.output_file, "w", encoding="utf-8") as output_file:
        for scenario in tqdm(scenarios, desc="Generating validation rollouts"):
            inferred_emotion = infer_emotion(
                model, tokenizer, scenario["problem"], scenario["situation"]
            )
            seed_info = {
                "problem": scenario["problem"],
                "emotion": inferred_emotion,
                "situation": scenario["situation"],
            }
            num_turns = random.randint(args.min_turns, args.max_turns)
            transcript = simulate_conversation(
                model,
                tokenizer,
                seed_info,
                num_turns,
                use_sft_seeker=True,
            )
            output_file.write(json.dumps({
                "id": scenario["source_index"],
                "source_index": scenario["source_index"],
                "num_turns": num_turns,
                "problem": scenario["problem"],
                "situation": scenario["situation"],
                "inferred_emotion": inferred_emotion,
                "reference_emotion": scenario["reference_emotion"],
                "sft_seeker_transcript": transcript,
            }, ensure_ascii=False) + "\n")
            output_file.flush()

    print(
        f"Saved {len(scenarios)} complete validation-seeded conversations "
        f"to {args.output_file}"
    )


if __name__ == "__main__":
    main()
