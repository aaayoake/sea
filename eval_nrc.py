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
    human_stats = (
        compute_nrc_stats([r["human_response"] for r in records], lexicon, lemmatizer)
        if has_human_reference
        else None
    )
    base_stats = compute_nrc_stats([r["base_response"] for r in records], lexicon, lemmatizer)
    sft_stats = compute_nrc_stats([r["sft_response"] for r in records], lexicon, lemmatizer)

    if human_stats:
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
        base_dice = sft_dice = None
        base_mean = sft_mean = None

    print(f"\n==================== {label} (n={len(records)}) ====================")
    if human_stats:
        print(
            f"Avg Words/Turn | Human: {human_stats['avg_words_per_turn']:.1f} | "
            f"Base: {base_stats['avg_words_per_turn']:.1f} | "
            f"SFT: {sft_stats['avg_words_per_turn']:.1f}"
        )
        print(
            f"{'Category':<14} {'Human %':>9} {'Base %':>9} {'SFT %':>9} | "
            f"{'Base Dice':>10} {'SFT Dice':>10}"
        )
    else:
        print(
            f"Avg Words/Turn | Base seeker: {base_stats['avg_words_per_turn']:.1f} | "
            f"SFT seeker: {sft_stats['avg_words_per_turn']:.1f}"
        )
        print(f"{'Category':<14} {'Base %':>9} {'SFT %':>9}")
    print("-" * (70 if human_stats else 38))
    for c in NRC_CATEGORIES:
        if human_stats:
            print(
                f"{c:<14} {human_stats['freqs'][c]*100:>8.2f}% "
                f"{base_stats['freqs'][c]*100:>8.2f}% "
                f"{sft_stats['freqs'][c]*100:>8.2f}% | "
                f"{base_dice[c]:>9.2f} {sft_dice[c]:>9.2f}"
            )
        else:
            print(f"{c:<14} {base_stats['freqs'][c]*100:>8.2f}% {sft_stats['freqs'][c]*100:>8.2f}%")
    print("-" * (70 if human_stats else 38))
    if human_stats:
        print(f"{'MEAN DICE':<44} | {base_mean:>9.2f} {sft_mean:>9.2f}")

    report = {
        "n": len(records),
        "avg_words_per_turn": {
            "human": human_stats["avg_words_per_turn"] if human_stats else None,
            "base": base_stats["avg_words_per_turn"],
            "sft": sft_stats["avg_words_per_turn"],
        },
        "base_mean_dice_vs_human": base_mean,
        "sft_mean_dice_vs_human": sft_mean,
        "per_category_dice_vs_human": (
            {"base": base_dice, "sft": sft_dice} if human_stats else None
        ),
        "emotion_frequencies": {
            "base": base_stats["freqs"],
            "sft": sft_stats["freqs"],
            **({"human": human_stats["freqs"]} if human_stats else {}),
        },
        "top_matched_words": {
            "human": human_stats["top_matched_words"] if human_stats else [],
            "base": base_stats["top_matched_words"],
            "sft": sft_stats["top_matched_words"],
        },
    }
    return report


def normalize_live_rollouts(source_records: list[dict]) -> list[dict]:
    records = []
    for record in source_records:
        if "sft_seeker_transcript" not in record:
            records.append(record)
            continue

        base_turns = [
            turn for turn in record["base_seeker_transcript"]
            if turn.get("speaker") == "seeker"
        ]
        sft_turns = [
            turn for turn in record["sft_seeker_transcript"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="eval_outputs/esconv_val_generations.jsonl")
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

    # Print Top-25 matched words for manual false-positive audit
    print("\n=== Top-25 NRC Matched Words (Spot-Check for Domain False Positives) ===")
    groups = ["base", "sft"]
    if all("human_response" in record for record in records):
        groups.insert(0, "human")
    for group in groups:
        words_str = ", ".join(f"{w}({c})" for w, c in report["overall"]["top_matched_words"][group])
        print(f"[{group.upper()}]: {words_str}\n")

    with open(args.report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Full JSON report written to {args.report_file}")


if __name__ == "__main__":
    main()