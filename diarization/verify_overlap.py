"""Verify speaker overlaps using Voxtral audio analysis.

This script detects overlapping speech regions and uses Voxtral to analyze
if they represent natural overlaps (backchanneling, agreement, excitement)
or unnatural interruptions (rude interruption, talking over someone).

Example:
    python diarization/verify_overlap.py \\
        --diarization output/convo_diarization.json \\
        --audio audio/convo.wav \\
        --threshold 0.5 \\
        --window 5.0
"""

import argparse
import json
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(
    0, str(Path(__file__).parent.parent / "model_usage" / "audio_understanding")
)

from overlap import (
    add_interruption_info,
    build_annotation,
    find_overlaps,
    load_diarization,
    merge_adjacent_overlaps,
)
from voxtral import analyze_audio_segment, load_voxtral_model


def verify_overlaps(
    overlaps: list[dict],
    audio_path: str,
    model,
    processor,
    device: str,
    window: float = 5.0,
) -> list[dict]:
    """Verify overlaps using Voxtral analysis.

    Args:
        overlaps: List of overlap events from find_overlaps()
        audio_path: Path to audio file
        model: Loaded Voxtral model
        processor: Loaded processor
        device: Device to run on
        window: Seconds of audio to include before/after the overlap

    Returns:
        List of verified events with Voxtral analysis
    """
    verified = []

    for i, overlap in enumerate(overlaps):
        speakers_str = " & ".join(overlap["speakers"])
        interruption = overlap.get("interruption")

        if interruption:
            context = (
                f"{interruption['interrupter']} interrupted "
                f"{', '.join(interruption['interrupted'])}"
            )
        else:
            context = "simultaneous speech"

        print(
            f"  Analyzing overlap {i + 1}/{len(overlaps)}: "
            f"{speakers_str} ({overlap['duration']:.2f}s at {overlap['start']:.1f}s) - {context}"
        )

        start_time = overlap["start"] - window
        end_time = overlap["end"] + window

        prompt = (
            "Listen to this audio segment carefully. There is an overlap where multiple speakers "
            "are talking at the same time. "
            "Is this a natural overlap (such as backchanneling like 'uh-huh', 'yeah', agreement, "
            "excitement, laughter, or natural conversation flow)? "
            "Or is this an unnatural/problematic overlap (such as a rude interruption, someone "
            "talking over another person, or disruptive speech)?\n\n"
            "Respond in the following format:\n"
            "ANALYSIS: <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the overlap feels natural or problematic>\n"
            "RESULT: <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified.append(
            {
                "start": overlap["start"],
                "end": overlap["end"],
                "duration": overlap["duration"],
                "speakers": overlap["speakers"],
                "interruption": interruption,
                "audio_window": {
                    "start": round(max(0, start_time), 3),
                    "end": round(end_time, 3),
                },
                "voxtral_analysis": analysis,
            }
        )

    return verified


def main():
    parser = argparse.ArgumentParser(
        description="Verify speaker overlaps using Voxtral audio analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diarization/verify_overlap.py --diarization output/convo.json --audio audio/convo.wav
  python diarization/verify_overlap.py --diarization output/convo.json --audio audio/convo.wav --threshold 0.5
  python diarization/verify_overlap.py --diarization output/convo.json --audio audio/convo.wav --merge -o verified.json
        """,
    )
    parser.add_argument(
        "--diarization",
        type=Path,
        required=True,
        help="Path to diarization JSON file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum overlap duration in seconds to analyze (default: 0.5)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Audio window in seconds to include before/after each overlap (default: 5.0)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge adjacent overlaps with the same speakers",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file for verified results (JSON)",
    )
    args = parser.parse_args()

    # Load diarization data
    print(f"Loading diarization from: {args.diarization}")
    segments = load_diarization(args.diarization)
    print(f"Found {len(segments)} segments")

    # Build annotation
    annotation = build_annotation(segments)
    speakers = annotation.labels()
    print(f"Speakers: {speakers}")

    # Find overlaps
    overlaps = find_overlaps(annotation)

    if args.merge:
        overlaps = merge_adjacent_overlaps(overlaps)
        print(f"Found {len(overlaps)} overlap regions (merged)")
    else:
        print(f"Found {len(overlaps)} overlap regions")

    # Add interruption classification
    overlaps = add_interruption_info(overlaps, annotation)

    # Filter by threshold
    overlaps = [o for o in overlaps if o["duration"] >= args.threshold]
    print(f"Overlaps above threshold ({args.threshold}s): {len(overlaps)}")

    if not overlaps:
        print(
            f"\nNo overlaps above threshold ({args.threshold}s) found. Nothing to verify."
        )
        return

    # Load Voxtral model
    model, processor, device = load_voxtral_model()

    # Verify overlaps
    print(f"\nVerifying {len(overlaps)} overlaps...")
    verified_overlaps = verify_overlaps(
        overlaps, args.audio, model, processor, device, args.window
    )

    # Count natural vs unnatural
    natural_count = 0
    unnatural_count = 0
    for overlap in verified_overlaps:
        analysis = overlap.get("voxtral_analysis", "").lower()
        if "result:" in analysis or "result :" in analysis:
            if "unnatural" in analysis.split("result")[-1]:
                unnatural_count += 1
            elif "natural" in analysis.split("result")[-1]:
                natural_count += 1

    # Print results
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    print(f"\nTotal overlaps analyzed: {len(verified_overlaps)}")
    print(f"Natural overlaps: {natural_count}")
    print(f"Unnatural overlaps: {unnatural_count}")

    print(f"\n--- OVERLAP DETAILS ({len(verified_overlaps)}) ---")
    for overlap in verified_overlaps:
        speakers_str = " & ".join(overlap["speakers"])
        interruption = overlap.get("interruption")
        if interruption:
            context = f"{interruption['interrupter']} -> {', '.join(interruption['interrupted'])}"
        else:
            context = "simultaneous"

        print(
            f"\n[{overlap['start']:.2f}s - {overlap['end']:.2f}s] "
            f"{speakers_str} ({overlap['duration']:.2f}s) - {context}"
        )
        print(f"  Analysis: {overlap['voxtral_analysis'][:200]}...")

    # Build output
    result = {
        "threshold": args.threshold,
        "window": args.window,
        "summary": {
            "total_overlaps": len(verified_overlaps),
            "natural_count": natural_count,
            "unnatural_count": unnatural_count,
        },
        "overlaps": verified_overlaps,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
