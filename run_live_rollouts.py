import argparse
import itertools
import json
import random
from pathlib import Path
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_esconv import (
    SEEKER_OPENING_PROMPT,
    build_supporter_system_prompt,
    build_system_prompt,
)

TOPIC_DETAILS = {
    "work": [
        "a recent change in responsibilities has left me unsure what is expected",
        "I am having trouble setting boundaries around work outside my normal hours",
        "feedback from someone at work has made me question how I am doing",
    ],
    "education": [
        "I am struggling to balance coursework with the rest of my life",
        "I am unsure whether the path I chose still fits me",
        "a recent setback has made it difficult to stay motivated",
    ],
    "friendship": [
        "I feel uncertain about where I stand with someone I care about",
        "a misunderstanding has made an important friendship feel different",
        "I have been putting effort into a friendship and am unsure it is mutual",
    ],
    "family": [
        "a disagreement at home keeps coming back without being resolved",
        "I feel caught between my own needs and what relatives expect",
        "communication with someone in my family has become tense lately",
    ],
    "finances": [
        "unexpected expenses have made it hard to feel secure",
        "I am trying to make a difficult choice about a major purchase",
        "money worries are affecting how I plan for the next few months",
    ],
    "health": [
        "uncertainty about a health concern is making it hard to focus",
        "I am adjusting to changes in my energy and daily routine",
        "I have been finding it difficult to ask for help with my wellbeing",
    ],
    "caregiving": [
        "I am supporting someone close to me and feel stretched thin",
        "I am trying to balance caregiving with responsibilities of my own",
        "I feel unsure how to get more support for someone who depends on me",
    ],
    "relationship": [
        "my partner and I have been struggling to understand each other's needs",
        "I am uncertain how to talk about a change I want in my relationship",
        "a recent disagreement has left me unsure how to reconnect",
    ],
    "moving": [
        "I am facing a move and feel unsettled about leaving familiar routines",
        "I am trying to make a new place feel more like home",
        "a possible change of location has made the future feel uncertain",
    ],
    "life direction": [
        "I am reconsidering a goal I have been working toward for a long time",
        "I feel pressure to make a decision about what comes next",
        "I am comparing my progress to other people and losing confidence",
    ],
    "self-confidence": [
        "I have been doubting myself after a few recent setbacks",
        "I find it hard to recognize things I am doing well",
        "I am nervous about taking a step that matters to me",
    ],
    "social connection": [
        "I have been feeling disconnected even when I am around other people",
        "I want to meet people but find it difficult to take the first step",
        "my usual routines have left me with fewer chances to connect",
    ],
}

EMOTIONS = (
    "anxiety", "sadness", "frustration", "loneliness",
    "overwhelm", "guilt", "uncertainty", "disappointment",
)

SCENARIO_FRAMINGS = (
    "Lately, {detail}.",
    "One thing on my mind is that {detail}.",
    "I have been trying to deal with this: {detail}.",
    "Recently, {detail}, and I have not been sure how to handle it.",
)


def build_scenario_pool(seed: int) -> list[dict]:
    scenarios = []
    for (topic, details), emotion, framing in itertools.product(
        TOPIC_DETAILS.items(), EMOTIONS, SCENARIO_FRAMINGS
    ):
        for detail in details:
            scenarios.append({
                "problem": topic,
                "emotion": emotion,
                "situation": framing.format(detail=detail),
            })
    random.Random(seed).shuffle(scenarios)
    return scenarios


def load_scenarios(scenario_file: str | None, seed: int) -> list[dict]:
    if scenario_file is None:
        return build_scenario_pool(seed)

    scenarios = []
    with open(scenario_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            scenario = json.loads(line)
            if not isinstance(scenario, dict):
                raise ValueError(f"{scenario_file}:{line_number} must contain a JSON object")
            missing = {"problem", "emotion", "situation"} - scenario.keys()
            if missing:
                raise ValueError(
                    f"{scenario_file}:{line_number} is missing fields: {', '.join(sorted(missing))}"
                )
            if not all(
                isinstance(scenario[key], str) and scenario[key].strip()
                for key in ("problem", "emotion", "situation")
            ):
                raise ValueError(
                    f"{scenario_file}:{line_number} fields problem, emotion, and situation "
                    "must be non-empty strings"
                )
            scenarios.append({
                key: scenario[key].strip()
                for key in ("problem", "emotion", "situation")
            })
    return scenarios


@torch.no_grad()
def generate_reply(model, tokenizer, messages: list[dict], max_new_tokens: int = 128) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def simulate_conversation(
    model,
    tokenizer,
    seed_info: dict,
    num_turns: int,
    use_sft_seeker: bool,
    max_new_tokens: int = 128,
) -> list[dict]:
    """Runs a complete multi-turn conversation between Seeker and Supporter."""
    seeker_messages = [
        {
            "role": "system",
            "content": build_system_prompt(
                seed_info["emotion"], seed_info["problem"], seed_info["situation"]
            ),
        },
        {"role": "user", "content": SEEKER_OPENING_PROMPT},
    ]
    supporter_messages = [{"role": "system", "content": build_supporter_system_prompt()}]
    transcript = []

    if use_sft_seeker:
        model.set_adapter("seeker")
        seeker_opening = generate_reply(model, tokenizer, seeker_messages, max_new_tokens)
    else:
        with model.disable_adapter():
            seeker_opening = generate_reply(model, tokenizer, seeker_messages, max_new_tokens)
    seeker_messages.append({"role": "assistant", "content": seeker_opening})
    supporter_messages.append({"role": "user", "content": seeker_opening})
    transcript.append({"speaker": "seeker", "content": seeker_opening})

    for _ in range(num_turns):
        # Each role receives only the scenario instructions and generated conversation so far.
        model.set_adapter("supporter")
        supporter_reply = generate_reply(model, tokenizer, supporter_messages, max_new_tokens)

        supporter_messages.append({"role": "assistant", "content": supporter_reply})
        seeker_messages.append({"role": "user", "content": supporter_reply})
        transcript.append({"speaker": "supporter", "content": supporter_reply})

        # --- 2. Seeker Turn (Either Fine-tuned Seeker LoRA or Base Llama) ---
        if use_sft_seeker:
            model.set_adapter("seeker")
            seeker_reply = generate_reply(model, tokenizer, seeker_messages, max_new_tokens)
        else:
            with model.disable_adapter():
                seeker_reply = generate_reply(model, tokenizer, seeker_messages, max_new_tokens)

        seeker_messages.append({"role": "assistant", "content": seeker_reply})
        supporter_messages.append({"role": "user", "content": seeker_reply})
        transcript.append({"speaker": "seeker", "content": seeker_reply})

    return transcript


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID; files must already exist in the configured local cache",
    )
    parser.add_argument("--seeker-adapter", type=str, default="checkpoints/esconv_seeker_sft/best")
    parser.add_argument("--supporter-adapter", type=str, default="checkpoints/esconv_supporter_sft/best")
    parser.add_argument(
        "--scenario-file",
        type=str,
        help="Optional JSONL of fresh scenarios with problem, emotion, and situation fields",
    )
    parser.add_argument("--output-file", type=str, default="eval_outputs/live_rollouts.jsonl")
    parser.add_argument("--num-conversations", type=int, default=50)
    parser.add_argument(
        "--min-turns",
        type=int,
        default=6,
        help="Minimum supporter/seeker exchange rounds after the seeker's opening",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum supporter/seeker exchange rounds after the seeker's opening",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    if args.min_turns < 1 or args.max_turns < args.min_turns:
        raise ValueError("Turn counts must satisfy 1 <= min-turns <= max-turns")
    if args.num_conversations < 1:
        raise ValueError("num-conversations must be positive")

    scenarios = load_scenarios(args.scenario_file, args.seed)
    if not scenarios:
        raise ValueError("No scenarios were provided or generated")
    if len(scenarios) < args.num_conversations:
        raise ValueError(
            f"Requested {args.num_conversations} conversations but only "
            f"{len(scenarios)} unique scenarios are available"
        )
    seeds = scenarios[: args.num_conversations]

    # 2. Load Base Llama ONCE and attach both LoRA adapters
    if not Path(args.seeker_adapter).is_dir():
        raise FileNotFoundError(f"Seeker adapter directory does not exist: {args.seeker_adapter}")
    if not Path(args.supporter_adapter).is_dir():
        raise FileNotFoundError(f"Supporter adapter directory does not exist: {args.supporter_adapter}")

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

    print("Attaching Seeker and Supporter LoRA adapters...")
    model = PeftModel.from_pretrained(
        base_model, args.seeker_adapter, adapter_name="seeker", local_files_only=True
    )
    model.load_adapter(
        args.supporter_adapter, adapter_name="supporter", local_files_only=True
    )
    model.eval()

    # 3. Run paired live rollouts (SFT seeker vs. base seeker), both with the supporter adapter.
    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for idx, seed_info in enumerate(tqdm(seeds, desc="Running live rollouts")):
            num_turns = random.randint(args.min_turns, args.max_turns)

            sft_transcript = simulate_conversation(
                model, tokenizer, seed_info, num_turns, use_sft_seeker=True
            )
            base_transcript = simulate_conversation(
                model, tokenizer, seed_info, num_turns, use_sft_seeker=False
            )

            record = {
                "id": idx,
                "num_turns": num_turns,
                "seed": seed_info,
                "sft_seeker_transcript": sft_transcript,
                "base_seeker_transcript": base_transcript,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(seeds)} paired full conversations to {args.output_file}")


if __name__ == "__main__":
    main()