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

Prepare the role-specific files once before training. Training jobs only read these files and check that the selected role's train and validation files exist; they do not rewrite them.

Submit one Slurm job per role:

```bash
SEEKER_JOB=$(sbatch --parsable sbatch/train_sft.sbatch seeker)
SUPPORTER_JOB=$(sbatch --parsable sbatch/train_sft.sbatch supporter)
echo "Seeker job: $SEEKER_JOB"
echo "Supporter job: $SUPPORTER_JOB"
```

Submit the jobs from the project directory (the directory containing `data/`). They use Slurm's `SLURM_SUBMIT_DIR` as the working directory. The scripts set `CONDA_PKGS_DIRS` to scratch and set `HF_HOME` to the course's shared, read-only model cache (`/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache`). They run Python directly from `$HOME/.conda/envs/sea/bin/python`, avoiding Conda activation in the batch shell. If the environment is elsewhere, set `ENV_PREFIX` when submitting. The cached model ID is `meta-llama/Llama-3.1-8B-Instruct` by default; override it with `MODEL_ID` if needed. Hugging Face/Transformers are in offline mode. Training uses SDPA attention, gradient checkpointing, a micro-batch of 1, and accumulation over 16 batches to reduce GPU memory use.

The seeker and supporter jobs read separate prepared datasets and write to separate checkpoint directories: `checkpoints/<job-id>/seeker/` and `checkpoints/<job-id>/supporter/`. This prevents a new training run from overwriting previous adapters, and the jobs can be submitted at the same time. Slurm may queue one if there is not another available GPU. Save the two job IDs printed above.

| Role | Training data | Adapter output |
|---|---|---|
| `seeker` | `data/esconv_sft_train.jsonl`, `data/esconv_sft_val.jsonl` | `checkpoints/<seeker-job-id>/seeker/` |
| `supporter` | `data/esconv_supporter_sft_train.jsonl`, `data/esconv_supporter_sft_val.jsonl` | `checkpoints/<supporter-job-id>/supporter/` |

## Generate complete conversations

After both training jobs complete successfully and their `best` adapters exist, pass their Slurm job IDs to evaluation:

```bash
sbatch sbatch/eval_nrc.sbatch "$SEEKER_JOB" "$SUPPORTER_JOB"
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
