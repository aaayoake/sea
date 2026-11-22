import argparse
import json
import random
from pathlib import Path
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def generate_reply(model, tokenizer, prompt_messages: list[dict], max_new_tokens: int) -> str:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
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
    generated_ids = outputs[0][input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID; files must already exist in the configured local cache",
    )
    parser.add_argument("--adapter-dir", type=str, default="checkpoints/esconv_seeker_sft/best")
    parser.add_argument("--val-file", type=str, default="data/esconv_sft_val.jsonl")
    parser.add_argument("--output-file", type=str, default="eval_outputs/esconv_val_generations.jsonl")
    parser.add_argument("--max-samples", type=int, default=500, help="Subsample val turns for fast eval (0 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    # Load held-out validation examples
    with open(args.val_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.max_samples > 0 and len(records) > args.max_samples:
        records = random.sample(records, args.max_samples)

    if not Path(args.adapter_dir).is_dir():
        raise FileNotFoundError(f"Adapter directory does not exist: {args.adapter_dir}")

    print(f"Loading cached base model ({args.model_id}) and LoRA adapter ({args.adapter_dir})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        base_model, args.adapter_dir, local_files_only=True
    )
    model.eval()

    with open(args.output_file, "w", encoding="utf-8") as out_f:
        for rec in tqdm(records, desc="Generating Base & SFT responses"):
            prompt_messages = rec["messages"][:-1]
            human_reply = rec["messages"][-1]["content"].strip()

            # 1. LoRA SFT simulator response
            sft_reply = generate_reply(model, tokenizer, prompt_messages, args.max_new_tokens)

            # 2. Zero-shot Base Llama response (adapter disabled)
            with model.disable_adapter():
                base_reply = generate_reply(model, tokenizer, prompt_messages, args.max_new_tokens)

            result = {
                "emotion": rec.get("emotion", "Unknown"),
                "problem": rec.get("problem", "Unknown"),
                "human_response": human_reply,
                "base_response": base_reply,
                "sft_response": sft_reply,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} paired evaluation responses to {args.output_file}")


if __name__ == "__main__":
    main()