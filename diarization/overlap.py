"""Analyze speaker overlap from diarization output using pyannote.core."""

import argparse
import json
from pathlib import Path

from pyannote.core import Annotation, Segment


def load_diarization(file_path: str | Path) -> list[dict]:
    """Load diarization data from JSON file."""
    with open(file_path) as f:
        data = json.load(f)
    return data.get("diarization", data)


def build_annotation(segments: list[dict]) -> Annotation:
    """Build a pyannote Annotation from diarization segments."""
    records = (
        (Segment(seg["start"], seg["end"]), i, seg["speaker"])
        for i, seg in enumerate(segments)
    )
    return Annotation.from_records(records)


def find_overlaps(annotation: Annotation) -> list[dict]:
    """Find all overlapping regions where multiple speakers are active.

    Returns a list of overlap events with:
    - start: start time of overlap
    - end: end time of overlap
    - duration: length of overlap in seconds
    - speakers: list of speakers involved
    """
    # Use get_overlap() to find all overlapping regions
    overlap_timeline = annotation.get_overlap()

    overlaps = []
    for segment in overlap_timeline:
        # Get labels active during this overlap segment
        cropped = annotation.crop(segment, mode="intersection")
        speakers = sorted(set(cropped.labels()))

        overlaps.append(
            {
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
                "speakers": speakers,
            }
        )

    return overlaps


def merge_adjacent_overlaps(overlaps: list[dict]) -> list[dict]:
    """Merge adjacent overlap regions with the same speakers."""
    if not overlaps:
        return []

    merged = [overlaps[0].copy()]

    for overlap in overlaps[1:]:
        prev = merged[-1]
        # Check if adjacent and same speakers
        if (
            abs(overlap["start"] - prev["end"]) < 0.001
            and overlap["speakers"] == prev["speakers"]
        ):
            # Extend the previous overlap
            prev["end"] = overlap["end"]
            prev["duration"] = prev["end"] - prev["start"]
        else:
            merged.append(overlap.copy())

    return merged


def classify_interruption(overlap: dict, annotation: Annotation) -> dict | None:
    """Determine who interrupted whom in an overlap.

    Returns None if both speakers started simultaneously (not a clear interruption).
    """
    overlap_start = overlap["start"]
    speakers = overlap["speakers"]

    interrupters = []
    interrupted = []

    for segment, track in annotation.itertracks():
        speaker = annotation[segment, track]
        if speaker not in speakers:
            continue

        # Check if segment is active during overlap
        if segment.start <= overlap_start < segment.end:
            if segment.start == overlap_start:
                interrupters.append(speaker)
            else:
                interrupted.append(speaker)

    # Skip if simultaneous start (both are interrupters or ambiguous)
    if len(interrupters) != 1 or len(interrupted) < 1:
        return None

    return {
        "interrupter": interrupters[0],
        "interrupted": list(set(interrupted)),
    }


def add_interruption_info(overlaps: list[dict], annotation: Annotation) -> list[dict]:
    """Add interruption classification to each overlap."""
    for overlap in overlaps:
        overlap["interruption"] = classify_interruption(overlap, annotation)
    return overlaps


def compute_statistics(overlaps: list[dict], annotation: Annotation) -> dict:
    """Compute overlap and interruption statistics."""
    if not overlaps:
        return {
            "total_overlap_duration": 0.0,
            "overlap_count": 0,
            "avg_overlap_duration": 0.0,
            "max_overlap_duration": 0.0,
            "total_speech_duration": 0.0,
            "overlap_percentage": 0.0,
            "speaker_pair_counts": {},
            "interruption_count": 0,
            "simultaneous_count": 0,
            "interruptions_by_speaker": {},
            "interrupted_by_speaker": {},
        }

    total_overlap = sum(o["duration"] for o in overlaps)
    max_overlap = max(o["duration"] for o in overlaps)

    # Count speaker pair occurrences
    pair_counts: dict[str, float] = {}
    for overlap in overlaps:
        speakers = overlap["speakers"]
        pair_key = " & ".join(speakers)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0.0) + overlap["duration"]

    # Calculate total speech duration using get_timeline().support()
    total_speech = annotation.get_timeline().support().duration()

    overlap_pct = (total_overlap / total_speech * 100) if total_speech > 0 else 0.0

    # Count interruptions per speaker
    interruptions_by: dict[str, int] = {}  # who interrupts
    interrupted_by: dict[str, int] = {}  # who gets interrupted
    interruption_count = 0
    simultaneous_count = 0

    for overlap in overlaps:
        interruption = overlap.get("interruption")
        if interruption is None:
            simultaneous_count += 1
        else:
            interruption_count += 1
            interrupter = interruption["interrupter"]
            interruptions_by[interrupter] = interruptions_by.get(interrupter, 0) + 1
            for speaker in interruption["interrupted"]:
                interrupted_by[speaker] = interrupted_by.get(speaker, 0) + 1

    return {
        "total_overlap_duration": round(total_overlap, 3),
        "overlap_count": len(overlaps),
        "avg_overlap_duration": round(total_overlap / len(overlaps), 3),
        "max_overlap_duration": round(max_overlap, 3),
        "total_speech_duration": round(total_speech, 3),
        "overlap_percentage": round(overlap_pct, 2),
        "speaker_pair_counts": {
            k: round(v, 3) for k, v in sorted(pair_counts.items(), key=lambda x: -x[1])
        },
        "interruption_count": interruption_count,
        "simultaneous_count": simultaneous_count,
        "interruptions_by_speaker": dict(
            sorted(interruptions_by.items(), key=lambda x: -x[1])
        ),
        "interrupted_by_speaker": dict(
            sorted(interrupted_by.items(), key=lambda x: -x[1])
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze speaker overlap in diarization output"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to diarization JSON file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file for overlap analysis (JSON)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge adjacent overlaps with the same speakers",
    )
    args = parser.parse_args()

    # Load and process
    print(f"Loading diarization from: {args.input_file}")
    segments = load_diarization(args.input_file)
    print(f"Found {len(segments)} segments")

    # Build annotation
    annotation = build_annotation(segments)
    print(f"Speakers: {annotation.labels()}")

    # Find overlaps
    overlaps = find_overlaps(annotation)

    if args.merge:
        overlaps = merge_adjacent_overlaps(overlaps)
        print(f"Found {len(overlaps)} overlap regions (merged)")
    else:
        print(f"Found {len(overlaps)} overlap regions")

    # Add interruption classification to each overlap
    overlaps = add_interruption_info(overlaps, annotation)

    # Compute statistics
    stats = compute_statistics(overlaps, annotation)

    # Print summary
    print("\n" + "=" * 50)
    print("OVERLAP ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total speech duration:    {stats['total_speech_duration']:.3f}s")
    print(f"Total overlap duration:   {stats['total_overlap_duration']:.3f}s")
    print(f"Overlap percentage:       {stats['overlap_percentage']:.2f}%")
    print(f"Number of overlap events: {stats['overlap_count']}")
    if stats["overlap_count"] > 0:
        print(f"Average overlap duration: {stats['avg_overlap_duration']:.3f}s")
        print(f"Max overlap duration:     {stats['max_overlap_duration']:.3f}s")

    if stats["speaker_pair_counts"]:
        print("\nOverlap by speaker pairs:")
        for pair, duration in stats["speaker_pair_counts"].items():
            print(f"  {pair}: {duration:.3f}s")

    # Print interruption analysis
    print("\n" + "-" * 50)
    print("INTERRUPTION ANALYSIS")
    print("-" * 50)
    print(f"Clear interruptions:      {stats['interruption_count']}")
    print(f"Simultaneous starts:      {stats['simultaneous_count']} (excluded)")

    if stats["interruptions_by_speaker"]:
        print("\nInterruptions made (who interrupts others):")
        for speaker, count in stats["interruptions_by_speaker"].items():
            print(f"  {speaker}: {count}")

    if stats["interrupted_by_speaker"]:
        print("\nTimes interrupted (who gets interrupted):")
        for speaker, count in stats["interrupted_by_speaker"].items():
            print(f"  {speaker}: {count}")

    # Output results
    result = {
        "statistics": stats,
        "overlaps": overlaps,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
