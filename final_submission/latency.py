"""Analyze speaker response latency from diarization output using pyannote.core."""

import json
from collections import defaultdict
from pathlib import Path

from pyannote.core import Annotation, Segment, Timeline


def load_diarization(file_path: str | Path) -> list[dict]:
    """Load diarization data from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return data.get("diarization", data)


def load_diarization_from_dict(data: dict) -> list[dict]:
    """Load diarization data from a dictionary."""
    return data.get("diarization", data)


def build_annotation(segments: list[dict]) -> Annotation:
    """Build a pyannote Annotation from diarization segments."""
    records = (
        (Segment(seg["start"], seg["end"]), i, seg["speaker"])
        for i, seg in enumerate(segments)
    )
    return Annotation.from_records(records)


def build_speaker_timelines(annotation: Annotation) -> dict[str, Timeline]:
    """Build a Timeline for each speaker from the annotation."""
    speaker_timelines: dict[str, Timeline] = {}
    for label in annotation.labels():
        speaker_timelines[label] = annotation.label_timeline(label)
    return speaker_timelines


def find_turn_transitions(annotation: Annotation) -> list[dict]:
    """Find all turn transitions between speakers.

    A turn transition occurs when one speaker's segment ends and another
    speaker's segment starts (the next segment in chronological order).

    Returns a list of transitions with:
    - prev_speaker: speaker who just finished
    - prev_end: end time of previous speaker's segment
    - next_speaker: speaker who starts next
    - next_start: start time of next speaker's segment
    - gap: time between prev_end and next_start (negative if overlap)
    """
    # Collect all segments with their speakers
    segments_with_speakers = []
    for segment, track in annotation.itertracks():
        speaker = annotation[segment, track]
        segments_with_speakers.append(
            {
                "segment": segment,
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end,
            }
        )

    # Sort by start time, then by end time
    segments_with_speakers.sort(key=lambda x: (x["start"], x["end"]))

    transitions = []
    for i in range(len(segments_with_speakers) - 1):
        current = segments_with_speakers[i]
        next_seg = segments_with_speakers[i + 1]

        # Only consider transitions between different speakers
        if current["speaker"] != next_seg["speaker"]:
            gap = next_seg["start"] - current["end"]
            transitions.append(
                {
                    "prev_speaker": current["speaker"],
                    "prev_end": current["end"],
                    "next_speaker": next_seg["speaker"],
                    "next_start": next_seg["start"],
                    "gap": gap,
                }
            )

    return transitions


def filter_clean_transitions(
    transitions: list[dict],
    annotation: Annotation,
    min_gap: float = 0.0,
    max_gap: float | None = None,
) -> list[dict]:
    """Filter transitions to only include clean, non-overlapping ones.

    A clean transition is one where:
    - The gap is non-negative (no overlap, gap >= 0)
    - No other speaker is active during the gap
    - Optionally, gap is within [min_gap, max_gap]

    Args:
        transitions: List of transition events
        annotation: The full annotation for checking activity
        min_gap: Minimum gap to consider (default 0.0)
        max_gap: Maximum gap to consider (None = no limit)

    Returns:
        Filtered list of clean transitions
    """
    clean = []

    for trans in transitions:
        gap = trans["gap"]

        # Skip overlapping transitions (negative gap)
        if gap < min_gap:
            continue

        # Skip if gap exceeds max (if specified)
        if max_gap is not None and gap > max_gap:
            continue

        # Check if any other speaker is active during the gap
        if gap > 0:
            gap_segment = Segment(trans["prev_end"], trans["next_start"])
            active_during_gap = annotation.crop(gap_segment, mode="intersection")
            active_speakers = set(active_during_gap.labels())

            # Remove the two speakers involved in the transition
            other_speakers = active_speakers - {
                trans["prev_speaker"],
                trans["next_speaker"],
            }

            # Skip if another speaker is active during the gap
            if other_speakers:
                continue

        clean.append(trans)

    return clean


def find_intra_speaker_pauses(
    annotation: Annotation,
    threshold: float = 4.0,
) -> list[dict]:
    """Find pauses within a single speaker's timeline when no one else is talking.

    Detects gaps between consecutive segments of the same speaker that
    exceed the threshold AND where no other speaker is active during the gap.
    These may indicate model failures or hesitations.

    Args:
        annotation: The full annotation
        threshold: Minimum pause duration to flag (default 1.0s)

    Returns:
        List of pause events with speaker, start, end, duration
    """
    pauses = []
    speaker_timelines = build_speaker_timelines(annotation)

    for speaker, timeline in speaker_timelines.items():
        # Get gaps in this speaker's timeline
        gaps = timeline.gaps()

        for gap in gaps:
            if gap.duration >= threshold:
                # Check if any other speaker is active during this gap
                gap_segment = Segment(gap.start, gap.end)
                active_during_gap = annotation.crop(gap_segment, mode="intersection")
                active_speakers = set(active_during_gap.labels())

                # Remove the current speaker (they're not active, it's their gap)
                other_speakers = active_speakers - {speaker}

                # Only flag if no other speaker is talking during this gap
                if not other_speakers:
                    pauses.append(
                        {
                            "speaker": speaker,
                            "start": round(gap.start, 3),
                            "end": round(gap.end, 3),
                            "duration": round(gap.duration, 3),
                            "type": "intra_speaker_pause",
                        }
                    )

    # Sort by start time
    pauses.sort(key=lambda x: x["start"])
    return pauses


def flag_slow_responses(
    transitions: list[dict],
    threshold: float = 2.0,
) -> list[dict]:
    """Flag transitions where the responding speaker took too long.

    Args:
        transitions: List of clean transition events
        threshold: Latency threshold to flag (default 1.0s)

    Returns:
        List of slow response events
    """
    slow = []
    for trans in transitions:
        if trans["gap"] >= threshold:
            slow.append(
                {
                    "prev_speaker": trans["prev_speaker"],
                    "prev_end": round(trans["prev_end"], 3),
                    "next_speaker": trans["next_speaker"],
                    "next_start": round(trans["next_start"], 3),
                    "latency": round(trans["gap"], 3),
                    "type": "slow_response",
                }
            )

    # Sort by latency (descending)
    slow.sort(key=lambda x: -x["latency"])
    return slow


def compute_latency_statistics(transitions: list[dict]) -> dict:
    """Compute latency statistics per responding speaker.

    Groups transitions by the responding speaker (next_speaker) and
    calculates aggregate statistics.

    Returns:
        Dictionary with per-speaker and overall statistics
    """
    if not transitions:
        return {
            "overall": {
                "avg_latency": 0.0,
                "min_latency": 0.0,
                "max_latency": 0.0,
                "response_count": 0,
            },
            "by_speaker": {},
            "by_speaker_pair": {},
        }

    # Group by responding speaker
    by_speaker: dict[str, list[float]] = defaultdict(list)
    by_pair: dict[str, list[float]] = defaultdict(list)

    for trans in transitions:
        latency = trans["gap"]
        by_speaker[trans["next_speaker"]].append(latency)
        pair_key = f"{trans['prev_speaker']} -> {trans['next_speaker']}"
        by_pair[pair_key].append(latency)

    # Calculate per-speaker statistics
    speaker_stats = {}
    for speaker, latencies in by_speaker.items():
        speaker_stats[speaker] = {
            "avg_latency": round(sum(latencies) / len(latencies), 3),
            "min_latency": round(min(latencies), 3),
            "max_latency": round(max(latencies), 3),
            "response_count": len(latencies),
        }

    # Calculate per-pair statistics
    pair_stats = {}
    for pair, latencies in by_pair.items():
        pair_stats[pair] = {
            "avg_latency": round(sum(latencies) / len(latencies), 3),
            "min_latency": round(min(latencies), 3),
            "max_latency": round(max(latencies), 3),
            "response_count": len(latencies),
        }

    # Calculate overall statistics
    all_latencies = [t["gap"] for t in transitions]
    overall = {
        "avg_latency": round(sum(all_latencies) / len(all_latencies), 3),
        "min_latency": round(min(all_latencies), 3),
        "max_latency": round(max(all_latencies), 3),
        "response_count": len(all_latencies),
    }

    return {
        "overall": overall,
        "by_speaker": dict(sorted(speaker_stats.items())),
        "by_speaker_pair": dict(
            sorted(pair_stats.items(), key=lambda x: -x[1]["response_count"])
        ),
    }
