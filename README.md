# ESConv Dual-Adapter Training and Fresh Conversation Rollouts

This project fine-tunes two LoRA adapters on ESConv using one shared LLaMA base model:

- **Seeker adapter:** simulates a person seeking emotional support.
- **Supporter adapter:** responds as a compassionate peer supporter.

ESConv is used as role-specific SFT data. The training examples include preceding dialogue turns so each role learns to respond in context. At rollout time, the prompts do **not** contain an ESConv transcript: the seeker starts from a fresh scenario and the two roles alternate using their newly generated conversation as context.

## Prepare training data

The cluster jobs expect the already-downloaded `esconv_train.json` file in the project directory. `download_datasets.py` is only for obtaining that file on a machine that can access Hugging Face; it is not run by the Slurm jobs.

```bash
python download_datasets.py
```

Generate both role-specific datasets:

```bash
python prepare_datasets.py
python prepare_esconv_supporter.py
```

Both preprocessors use the same deterministic dialogue-level train/validation split. The seeker dataset targets `usr` turns; the supporter dataset targets `sys` turns. The prompts contain role instructions and prior conversation context, but no conversation-stage labels.

## Train adapters

Submit one Slurm job per role:

```bash
MODEL_PATH=/path/to/local/base-model sbatch sbatch/train_sft.sbatch seeker
MODEL_PATH=/path/to/local/base-model sbatch sbatch/train_sft.sbatch supporter
```

Set `MODEL_PATH` to the cluster-local directory containing the base model configuration, weights, tokenizer, and associated files. The batch scripts set Hugging Face/Transformers offline mode and pass this local path to the loader; they will not try to download the model. The script regenerates both role datasets from local `esconv_train.json`, validates the requested role, and trains that role's adapter using the custom PyTorch loop in `train_esconv_sft.py`.

| Role | Training data | Adapter output |
|---|---|---|
| `seeker` | `data/esconv_sft_train.jsonl`, `data/esconv_sft_val.jsonl` | `checkpoints/esconv_seeker_sft/` |
| `supporter` | `data/esconv_supporter_sft_train.jsonl`, `data/esconv_supporter_sft_val.jsonl` | `checkpoints/esconv_supporter_sft/` |

## Generate complete conversations

After both `best` adapters exist:

```bash
python run_live_rollouts.py \
  --model-path /path/to/local/base-model \
  --seeker-adapter checkpoints/esconv_seeker_sft/best \
  --supporter-adapter checkpoints/esconv_supporter_sft/best \
  --num-conversations 50 \
  --min-turns 8 \
  --max-turns 12
```

By default, the script creates fresh scenario seeds by combining topics, emotions, and original scenario descriptions defined in the rollout script. They are not selected from ESConv train or validation dialogues. The seeker adapter generates the opening message in its own words, then the supporter and seeker take turns. Each role receives the scenario instructions relevant to it and only the conversation generated so far. The supporter does not receive the seeker's private scenario/backstory, just as a real agent would not.

`--min-turns` and `--max-turns` count supporter/seeker exchange rounds after the initial seeker opening. Thus, six rounds produce one seeker opening plus six supporter replies and six seeker replies.

To supply your own scenario seeds, pass `--scenario-file path/to/scenarios.jsonl`. Each non-empty line must be a JSON object with non-empty string fields `problem`, `emotion`, and `situation`. For example:

```json
{"problem":"work","emotion":"uncertainty","situation":"A recent change in responsibilities has left me unsure what is expected."}
```

The output contains paired transcripts: the supporter adapter is used in both, while the seeker is generated once with the seeker adapter and once with the unadapted base model. This allows comparison of seeker fine-tuning while holding the supporter constant.

## Evaluate

The NRC evaluation reads the paired live transcripts and compares emotion-word frequencies and word counts for base-seeker and SFT-seeker responses. Because synthetic conversations do not have a human response target for each turn, this comparison does not claim human alignment or calculate human-reference Dice scores.

## Dependencies and runtime

Install packages from `requirements.txt` in the cluster environment. The Slurm scripts assume jobs are submitted from the project directory, the local dataset and model directory are present, and training/rollout have a CUDA-capable GPU.
# sea
# sea
