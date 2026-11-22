import argparse
import json
import math
import os
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return Dataset.from_list(records)


def tokenize_with_response_mask(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    """
    Tokenizes a multi-turn conversation using the Llama chat template and creates a binary
    response_mask where ONLY the final target assistant turn has mask == 1.
    """
    assert messages[-1]["role"] == "assistant", "Final target turn must have the assistant role."

    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )["input_ids"]

    shared_length = min(len(prompt_ids), len(full_ids))
    if prompt_ids[:shared_length] != full_ids[:shared_length]:
        raise ValueError(
            "Chat-template tokenization does not preserve the prompt as a prefix of "
            "the full conversation; cannot safely construct the response loss mask."
        )

    prompt_len = min(len(prompt_ids), len(full_ids))
    response_mask = [0] * prompt_len + [1] * (len(full_ids) - prompt_len)
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.long),
    }


def build_sft_example(messages: list[dict], tokenizer, max_seq_len: int) -> dict:
    example = tokenize_with_response_mask(messages, tokenizer, max_seq_len)
    labels = example["input_ids"].clone()
    labels[example["response_mask"] == 0] = -100
    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"],
        "labels": labels,
    }


def build_model_and_tokenizer(model_id: str, rank: int, alpha: int, dropout: float):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.config.use_cache = False

    # Target all linear attention & MLP projections for strong behavioral adaptation
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


@torch.no_grad()
def evaluate(model, val_dataloader) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in val_dataloader:
        outputs = model(**batch)
        if not torch.isnan(outputs.loss):
            total_loss += outputs.loss.item()
            steps += 1
    model.train()
    return total_loss / max(steps, 1)


def train_loop(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    grad_accum_steps: int,
    epochs: int,
    lr: float,
    max_seq_len: int,
    checkpoint_dir: str,
) -> None:
    print("Tokenizing train and validation datasets...")
    processed_train = train_dataset.map(
        lambda ex: build_sft_example(ex["messages"], tokenizer, max_seq_len),
        remove_columns=train_dataset.column_names,
        batched=False,
    )
    processed_val = val_dataset.map(
        lambda ex: build_sft_example(ex["messages"], tokenizer, max_seq_len),
        remove_columns=val_dataset.column_names,
        batched=False,
    )

    # Filter out any truncated examples where the entire assistant response was cut off
    processed_train = processed_train.filter(lambda ex: any(l != -100 for l in ex["labels"]))
    processed_val = processed_val.filter(lambda ex: any(l != -100 for l in ex["labels"]))

    def collate_fn(batch):
        input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        padded_attention = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": padded_inputs.to(model.device),
            "attention_mask": padded_attention.to(model.device),
            "labels": padded_labels.to(model.device),
        }

    train_loader = torch.utils.data.DataLoader(
        processed_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        processed_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    total_update_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_update_steps),
        num_training_steps=total_update_steps,
    )

    best_val_loss = float("inf")
    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Step {step}/{len(train_loader)} "
                    f"| Train Loss: {loss.item() * grad_accum_steps:.4f}"
                )

        val_loss = evaluate(model, val_loader)
        print(f"--> End of Epoch {epoch + 1} | Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = os.path.join(checkpoint_dir, "best")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"Saved new best adapter to {best_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID; files must already exist in the configured local cache",
    )
    parser.add_argument("--role", choices=("seeker", "supporter"), default="seeker")
    parser.add_argument("--train-file", type=str)
    parser.add_argument("--val-file", type=str)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)  # Effective batch size = 16
    parser.add_argument("--epochs", type=int, default=2)      # 2 epochs prevents backstory memorization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    default_data_prefix = "esconv" if args.role == "seeker" else "esconv_supporter"
    args.train_file = args.train_file or f"data/{default_data_prefix}_sft_train.jsonl"
    args.val_file = args.val_file or f"data/{default_data_prefix}_sft_val.jsonl"
    args.checkpoint_dir = args.checkpoint_dir or f"checkpoints/esconv_{args.role}_sft"
    if args.batch_size < 1 or args.grad_accum < 1 or args.epochs < 1:
        parser.error("batch-size, grad-accum, and epochs must all be positive")

    torch.manual_seed(args.seed)

    model, tokenizer = build_model_and_tokenizer(
        args.model_id, args.rank, args.alpha, args.dropout
    )
    train_ds = load_jsonl(args.train_file)
    val_ds = load_jsonl(args.val_file)

    train_loop(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        checkpoint_dir=args.checkpoint_dir,
    )

    final_dir = os.path.join(args.checkpoint_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Training complete. Final adapter saved to {final_dir}")


if __name__ == "__main__":
    main()