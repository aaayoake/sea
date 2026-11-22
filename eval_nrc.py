import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

NRC_CATEGORIES = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative",
]


def load_nrc_lexicon(lexicon_path: str | None = None) -> dict[str, list[str]]:
    """
    Loads the NRC Emotion Lexicon as {word: [emotions...]}.
    Uses the official tab-separated .txt file if provided, otherwise falls back to NRCLex.
    """
    lexicon = defaultdict(list)

    if lexicon_path:
        path = Path(lexicon_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"NRC lexicon file not found or is not a file: {path}. "
                "Pass the official NRC-Emotion-Lexicon-Wordlevel-v0.92.txt file."
            )
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3 and parts[2] == "1":
                    word, emotion = parts[0].lower(), parts[1].lower()
                    if emotion in NRC_CATEGORIES:
                        lexicon[word].append(emotion)
        if not lexicon:
            raise ValueError(f"No valid NRC emotion entries found in lexicon file: {path}")
    else:
        from nrclex import NRCLex
        # NRCLex stores its internal word -> [emotions] mapping on the class/instance
        try:
            dummy = NRCLex("")
        except (IsADirectoryError, FileNotFoundError) as exc:
            raise RuntimeError(
                "NRCLex could not locate its bundled lexicon (it resolved a directory "
                "instead of a lexicon file). Pass the official NRC lexicon explicitly "
                "with --lexicon-path PATH or set NRC_LEXICON_PATH to that file."
            ) from exc
        raw_dict = getattr(dummy, "__lexicon__", None) or getattr(NRCLex, "__lexicon__", None)
        if raw_dict is None:
            raise RuntimeError(
                "Could not read internal NRCLex dictionary. "
                "Please download NRC-Emotion-Lexicon-Wordlevel-v0.92.txt and pass --lexicon-path."
            )
        for word, emotions in raw_dict.items():
            valid = [e.lower() for e in emotions if e.lower() in NRC_CATEGORIES]
            if valid:
                lexicon[word.lower()] = valid

    return dict(lexicon)


def _to_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_text(text: str, lemmatizer: WordNetLemmatizer) -> list[str]:
    tokens = [t.lower() for t in word_tokenize(text) if re.match(r"^[a-zA-Z']+$", t)]
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(tok, _to_wordnet_pos(pos)) for tok, pos in tagged]


def compute_nrc_stats(texts: list[str], lexicon: dict[str, list[str]], lemmatizer: WordNetLemmatizer) -> dict:
    total_words = 0
    emotion_counts = Counter({cat: 0 for cat in NRC_CATEGORIES})
    matched_words = Counter()

    for text in texts:
        lemmas = lemmatize_text(text, lemmatizer)
        total_words += len(lemmas)
        for lemma in lemmas:
            if lemma in lexicon:
                matched_words[lemma] += 1
                for emo in lexicon[lemma]:
                    emotion_counts[emo] += 1

    freqs = {
        cat: (emotion_counts[cat] / total_words) if total_words > 0 else 0.0
        for cat in NRC_CATEGORIES
    }
    avg_words_per_turn = total_words / max(len(texts), 1)

    return {
        "total_turns": len(texts),
        "total_words": total_words,
        "avg_words_per_turn": avg_words_per_turn,
        "freqs": freqs,
        "top_matched_words": matched_words.most_common(25),
    }


def sorensen_dice(model_val: float, human_val: float) -> float:
    if model_val == 0.0 and human_val == 0.0:
        return 100.0
    return (2.0 * min(model_val, human_val) / (model_val + human_val)) * 100.0


def evaluate_subset(records: list[dict], lexicon: dict, lemmatizer: WordNetLemmatizer, label: str) -> dict:
    has_human_reference = all("human_response" in record for record in records)
    has_base = all("base_response" in record for record in records)
    has_sft = all("sft_response" in record for record in records)
    if not has_sft and not has_base:
        raise ValueError("Evaluation records must contain sft_response or base_response")

    human_stats = (
        compute_nrc_stats([r["human_response"] for r in records], lexicon, lemmatizer)
        if has_human_reference
        else None
    )
    base_stats = (
        compute_nrc_stats([r["base_response"] for r in records], lexicon, lemmatizer)
        if has_base
        else None
    )
    sft_stats = (
        compute_nrc_stats([r["sft_response"] for r in records], lexicon, lemmatizer)
        if has_sft
        else None
    )

    if human_stats and base_stats and sft_stats:
        base_dice = {
            c: sorensen_dice(base_stats["freqs"][c], human_stats["freqs"][c])
            for c in NRC_CATEGORIES
        }
        sft_dice = {
            c: sorensen_dice(sft_stats["freqs"][c], human_stats["freqs"][c])
            for c in NRC_CATEGORIES
        }
        base_mean = sum(base_dice.values()) / len(NRC_CATEGORIES)
        sft_mean = sum(sft_dice.values()) / len(NRC_CATEGORIES)
    else:
        base_dice = (
            {
                c: sorensen_dice(base_stats["freqs"][c], human_stats["freqs"][c])
                for c in NRC_CATEGORIES
            }
            if human_stats and base_stats
            else None
        )
        sft_dice = (
            {
                c: sorensen_dice(sft_stats["freqs"][c], human_stats["freqs"][c])
                for c in NRC_CATEGORIES
            }
            if human_stats and sft_stats
            else None
        )
        base_mean = sum(base_dice.values()) / len(NRC_CATEGORIES) if base_dice else None
        sft_mean = sum(sft_dice.values()) / len(NRC_CATEGORIES) if sft_dice else None

    print(f"\n==================== {label} (n={len(records)}) ====================")
    if human_stats and base_stats and sft_stats:
        print(
            f"Avg Words/Turn | Human: {human_stats['avg_words_per_turn']:.1f} | "
            f"Base: {base_stats['avg_words_per_turn']:.1f} | "
            f"SFT: {sft_stats['avg_words_per_turn']:.1f}"
        )
        print(
            f"{'Category':<14} {'Human %':>9} {'Base %':>9} {'SFT %':>9} | "
            f"{'Base Dice':>10} {'SFT Dice':>10}"
        )
    elif base_stats and sft_stats:
        print(
            f"Avg Words/Turn | Base seeker: {base_stats['avg_words_per_turn']:.1f} | "
            f"SFT seeker: {sft_stats['avg_words_per_turn']:.1f}"
        )
        print(f"{'Category':<14} {'Base %':>9} {'SFT %':>9}")
    elif sft_stats:
        print(f"Avg Words/Turn | SFT seeker: {sft_stats['avg_words_per_turn']:.1f}")
        print(f"{'Category':<14} {'SFT %':>9}")
    else:
        print(f"Avg Words/Turn | Base seeker: {base_stats['avg_words_per_turn']:.1f}")
        print(f"{'Category':<14} {'Base %':>9}")
    print("-" * (70 if human_stats and base_stats and sft_stats else (38 if base_stats and sft_stats else 26)))
    for c in NRC_CATEGORIES:
        if human_stats and base_stats and sft_stats:
            print(
                f"{c:<14} {human_stats['freqs'][c]*100:>8.2f}% "
                f"{base_stats['freqs'][c]*100:>8.2f}% "
                f"{sft_stats['freqs'][c]*100:>8.2f}% | "
                f"{base_dice[c]:>9.2f} {sft_dice[c]:>9.2f}"
            )
        elif base_stats and sft_stats:
            print(f"{c:<14} {base_stats['freqs'][c]*100:>8.2f}% {sft_stats['freqs'][c]*100:>8.2f}%")
        elif sft_stats:
            print(f"{c:<14} {sft_stats['freqs'][c]*100:>8.2f}%")
        else:
            print(f"{c:<14} {base_stats['freqs'][c]*100:>8.2f}%")
    print("-" * (70 if human_stats and base_stats and sft_stats else (38 if base_stats and sft_stats else 26)))
    if human_stats and base_stats and sft_stats:
        print(f"{'MEAN DICE':<44} | {base_mean:>9.2f} {sft_mean:>9.2f}")
    elif human_stats and (base_dice or sft_dice):
        means = []
        if base_mean is not None:
            means.append(f"Base: {base_mean:.2f}")
        if sft_mean is not None:
            means.append(f"SFT: {sft_mean:.2f}")
        print(f"Mean Dice vs human | {' | '.join(means)}")

    report = {
        "n": len(records),
        "avg_words_per_turn": {
            "human": human_stats["avg_words_per_turn"] if human_stats else None,
            "base": base_stats["avg_words_per_turn"] if base_stats else None,
            "sft": sft_stats["avg_words_per_turn"] if sft_stats else None,
        },
        "base_mean_dice_vs_human": base_mean,
        "sft_mean_dice_vs_human": sft_mean,
        "per_category_dice_vs_human": (
            {"base": base_dice, "sft": sft_dice} if human_stats else None
        ),
        "emotion_frequencies": {
            **({"base": base_stats["freqs"]} if base_stats else {}),
            **({"sft": sft_stats["freqs"]} if sft_stats else {}),
            **({"human": human_stats["freqs"]} if human_stats else {}),
        },
        "top_matched_words": {
            "human": human_stats["top_matched_words"] if human_stats else [],
            **({"base": base_stats["top_matched_words"]} if base_stats else {}),
            **({"sft": sft_stats["top_matched_words"]} if sft_stats else {}),
        },
    }
    return report


def normalize_live_rollouts(source_records: list[dict]) -> list[dict]:
    records = []
    for record in source_records:
        if "sft_seeker_transcript" not in record:
            records.append(record)
            continue

        sft_turns = [
            turn for turn in record["sft_seeker_transcript"]
            if turn.get("speaker") == "seeker"
        ]
        if "base_seeker_transcript" not in record:
            records.extend(
                {"sft_response": turn["content"]}
                for turn in sft_turns
            )
        else:
            base_turns = [
                turn for turn in record["base_seeker_transcript"]
                if turn.get("speaker") == "seeker"
            ]
            if len(base_turns) != len(sft_turns):
                raise ValueError(f"Rollout {record.get('id', '?')} has unpaired seeker turns")
            for base_turn, sft_turn in zip(base_turns, sft_turns):
                records.append({
                    "base_response": base_turn["content"],
                    "sft_response": sft_turn["content"],
                })
    return records


def evaluate_validation_references(
    source_records: list[dict],
    reference_file: str,
    lexicon: dict,
    lemmatizer: WordNetLemmatizer,
) -> dict:
    with open(reference_file, "r", encoding="utf-8") as file:
        reference_dialogues = [
            json.loads(row["text"]) if isinstance(row.get("text"), str) else row
            for line in file
            if line.strip()
            for row in [json.loads(line)]
        ]

    generated_texts = []
    human_texts = []
    emotion_results = defaultdict(list)
    correct_emotions = 0
    seen_indices = set()

    for record in source_records:
        if "sft_seeker_transcript" not in record:
            raise ValueError(
                "Validation reference comparison requires sft_seeker_transcript records"
            )
        source_index = record.get("source_index")
        if not isinstance(source_index, int) or not 0 <= source_index < len(reference_dialogues):
            raise ValueError(
                f"Invalid or out-of-range source_index in rollout: {source_index!r}"
            )
        if source_index in seen_indices:
            raise ValueError(f"Duplicate validation source_index in rollouts: {source_index}")
        seen_indices.add(source_index)

        reference = reference_dialogues[source_index]
        if record.get("problem") != reference.get("problem_type"):
            raise ValueError(
                f"Rollout {source_index} problem does not match the validation reference"
            )
        if record.get("situation") != reference.get("situation"):
            raise ValueError(
                f"Rollout {source_index} situation does not match the validation reference"
            )

        generated_texts.extend(
            turn["content"]
            for turn in record["sft_seeker_transcript"]
            if turn.get("speaker") == "seeker" and turn.get("content", "").strip()
        )
        human_texts.extend(
            turn["text"].strip()
            for turn in reference.get("dialog", [])
            if turn.get("speaker") == "usr" and turn.get("text", "").strip()
        )

        predicted = record.get("inferred_emotion")
        expected = reference.get("emotion_type")
        if not isinstance(predicted, str) or not isinstance(expected, str):
            raise ValueError(
                f"Rollout {source_index} is missing inferred or reference emotion"
            )
        expected = expected.strip().lower()
        predicted = predicted.strip().lower()
        is_correct = predicted == expected
        emotion_results[expected].append(is_correct)
        correct_emotions += is_correct

    if not seen_indices:
        raise ValueError("No validation rollout records were provided for comparison")

    generated_stats = compute_nrc_stats(generated_texts, lexicon, lemmatizer)
    human_stats = compute_nrc_stats(human_texts, lexicon, lemmatizer)
    per_category_dice = {
        category: sorensen_dice(
            generated_stats["freqs"][category], human_stats["freqs"][category]
        )
        for category in NRC_CATEGORIES
    }

    per_emotion = {}
    for emotion in sorted(emotion_results):
        emotion_records = emotion_results[emotion]
        matched = sum(emotion_records)
        per_emotion[emotion] = {
            "n": len(emotion_records),
            "correct": matched,
            "accuracy": matched / len(emotion_records),
        }

    mean_dice = sum(per_category_dice.values()) / len(NRC_CATEGORIES)
    accuracy = correct_emotions / len(seen_indices)
    print(
        f"\nValidation reference comparison (n={len(seen_indices)}): "
        f"mean NRC Dice={mean_dice:.2f}, emotion-label accuracy={accuracy:.3f}"
    )
    return {
        "n_conversations": len(seen_indices),
        "n_generated_seeker_turns": generated_stats["total_turns"],
        "n_human_seeker_turns": human_stats["total_turns"],
        "nrc": {
            "generated_sft": generated_stats,
            "human_reference": human_stats,
            "per_category_dice": per_category_dice,
            "mean_dice": mean_dice,
        },
        "emotion_label_comparison": {
            "correct": correct_emotions,
            "accuracy": accuracy,
            "per_reference_emotion": per_emotion,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="eval_outputs/esconv_val_generations.jsonl")
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help=(
            "Optional ESConv validation JSONL. Matches source_index to compare generated "
            "seeker NRC profiles and inferred emotion against human references."
        ),
    )
    parser.add_argument(
        "--lexicon-path",
        type=str,
        default=os.environ.get("NRC_LEXICON_PATH"),
        help=(
            "Path to NRC-Emotion-Lexicon-Wordlevel-v0.92.txt "
            "(defaults to NRC_LEXICON_PATH when set)"
        ),
    )
    parser.add_argument("--report-file", type=str, default="eval_outputs/nrc_eval_report.json")
    args = parser.parse_args()

    try:
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("wordnet", quiet=True)
        lexicon = load_nrc_lexicon(args.lexicon_path)
        lemmatizer = WordNetLemmatizer()
    except LookupError as exc:
        raise RuntimeError(
            "Missing NLTK resource. Install the 'punkt_tab', "
            "'averaged_perceptron_tagger_eng', and 'wordnet' resources."
        ) from exc

    with open(args.input_file, "r", encoding="utf-8") as f:
        source_records = [json.loads(line) for line in f if line.strip()]

    records = normalize_live_rollouts(source_records)

    if not records:
        raise ValueError("The input file contains no evaluation responses")

    report = {"overall": evaluate_subset(records, lexicon, lemmatizer, "OVERALL")}
    if args.reference_file:
        report["validation_reference_comparison"] = evaluate_validation_references(
            source_records, args.reference_file, lexicon, lemmatizer
        )

    # Print Top-25 matched words for manual false-positive audit
    print("\n=== Top-25 NRC Matched Words (Spot-Check for Domain False Positives) ===")
    groups = list(report["overall"]["top_matched_words"])
    if all("human_response" in record for record in records):
        groups.remove("human")
        groups.insert(0, "human")
    for group in groups:
        words_str = ", ".join(f"{w}({c})" for w, c in report["overall"]["top_matched_words"][group])
        print(f"[{group.upper()}]: {words_str}\n")

    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Full JSON report written to {args.report_file}")


if __name__ == "__main__":
    main()