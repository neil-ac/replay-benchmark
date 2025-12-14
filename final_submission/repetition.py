"""Analyze consecutive n-gram repetitions within speaker segments."""

import json
import re
from collections import defaultdict
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline


def load_transcription(file_path: str | Path) -> list[dict]:
    """Load transcription data from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return data.get("turnLevelTranscription", data)


def load_transcription_from_dict(data: dict) -> list[dict]:
    """Load transcription data from a dictionary."""
    return data.get("turnLevelTranscription", data)


def normalize_word(word: str) -> str:
    """Normalize a single word: lowercase, strip punctuation."""
    return re.sub(r"[^\w]", "", word.lower())


def tokenize(text: str) -> list[tuple[str, str]]:
    """Tokenize text into (original_word, normalized_word) pairs."""
    words = text.split()
    return [(w, normalize_word(w)) for w in words if normalize_word(w)]


def find_consecutive_ngram_repetitions(
    tokens: list[tuple[str, str]],
    min_n: int = 1,
    max_n: int = 10,
) -> list[dict]:
    """Find consecutive repetitions of n-grams within a token sequence.

    For each n-gram size from min_n to max_n, scan through the tokens
    and find where the same n-gram appears consecutively.

    Returns a list of repetition events with:
    - n_gram_size: size of the n-gram
    - phrase: the original phrase (from first occurrence)
    - phrase_normalized: the normalized phrase
    - count: number of consecutive occurrences
    - start_token_idx: starting token index in the sequence
    - end_token_idx: ending token index (exclusive)
    """
    if not tokens:
        return []

    repetitions = []

    for n in range(min_n, min(max_n + 1, len(tokens) + 1)):
        if len(tokens) < n:
            continue

        # Scan through token sequence looking for consecutive n-gram repetitions
        i = 0
        while i <= len(tokens) - n:
            # Extract current n-gram (normalized for comparison)
            current_normalized = tuple(t[1] for t in tokens[i : i + n])
            current_original = " ".join(t[0] for t in tokens[i : i + n])

            # Count how many times this n-gram repeats consecutively
            count = 1
            j = i + n

            while j <= len(tokens) - n:
                next_normalized = tuple(t[1] for t in tokens[j : j + n])
                if next_normalized == current_normalized:
                    count += 1
                    j += n
                else:
                    break

            # Record if we found a repetition (count >= 2)
            if count >= 2:
                repetitions.append(
                    {
                        "n_gram_size": n,
                        "phrase": current_original,
                        "phrase_normalized": " ".join(current_normalized),
                        "count": count,
                        "start_token_idx": i,
                        "end_token_idx": j,
                    }
                )
                # Move past this entire repetition sequence
                i = j
            else:
                i += 1

    return repetitions


def build_speaker_timelines(
    turns: list[dict],
) -> dict[str, list[dict]]:
    """Build a mapping from speaker to their turns (with text).

    Returns dict mapping speaker -> list of turns with start, end, text.
    """
    speaker_turns: dict[str, list[dict]] = defaultdict(list)
    for turn in turns:
        speaker = turn["speaker"]
        speaker_turns[speaker].append(turn)
    return dict(speaker_turns)


def find_all_repetitions(
    turns: list[dict],
    min_n: int = 1,
    max_n: int = 10,
) -> list[dict]:
    """Find all consecutive n-gram repetitions across all speaker segments.

    For each speaker, for each of their segments, find consecutive
    n-gram repetitions within that segment's text.

    Returns list of repetition events.
    """
    speaker_turns = build_speaker_timelines(turns)
    all_repetitions = []

    for speaker, s_turns in speaker_turns.items():
        for turn in s_turns:
            text = turn.get("text", "")
            tokens = tokenize(text)

            if len(tokens) < 2:
                continue

            segment_repetitions = find_consecutive_ngram_repetitions(
                tokens, min_n, max_n
            )

            for rep in segment_repetitions:
                all_repetitions.append(
                    {
                        "speaker": speaker,
                        "segment_start": turn["start"],
                        "segment_end": turn["end"],
                        "segment_text": text,
                        "n_gram_size": rep["n_gram_size"],
                        "phrase": rep["phrase"],
                        "phrase_normalized": rep["phrase_normalized"],
                        "count": rep["count"],
                        "start_token_idx": rep["start_token_idx"],
                        "end_token_idx": rep["end_token_idx"],
                    }
                )

    # Sort by count (descending), then by n-gram size (descending)
    all_repetitions.sort(key=lambda x: (-x["count"], -x["n_gram_size"]))

    return all_repetitions


def compute_statistics(repetitions: list[dict], turns: list[dict]) -> dict:
    """Compute repetition statistics."""
    if not repetitions:
        speakers = set(t["speaker"] for t in turns)
        return {
            "total_repetitions": 0,
            "by_speaker": {s: 0 for s in speakers},
            "by_ngram_size": {},
            "most_repeated_phrase": None,
        }

    by_speaker: dict[str, int] = defaultdict(int)
    by_ngram_size: dict[int, int] = defaultdict(int)

    for rep in repetitions:
        by_speaker[rep["speaker"]] += 1
        by_ngram_size[rep["n_gram_size"]] += 1

    most_repeated = max(repetitions, key=lambda x: x["count"])

    return {
        "total_repetitions": len(repetitions),
        "by_speaker": dict(sorted(by_speaker.items(), key=lambda x: -x[1])),
        "by_ngram_size": dict(sorted(by_ngram_size.items())),
        "most_repeated_phrase": {
            "phrase": most_repeated["phrase"],
            "speaker": most_repeated["speaker"],
            "count": most_repeated["count"],
            "segment_text": most_repeated["segment_text"],
        },
    }
