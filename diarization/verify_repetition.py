"""Verify n-gram repetitions using Voxtral audio analysis.

This script detects consecutive n-gram repetitions and uses Voxtral to analyze
if they represent natural speech patterns or unnatural/problematic repetitions.

Example:
    python diarization/verify_repetition.py \\
        --transcription output/convo_transcription.json \\
        --audio audio/convo.wav \\
        --window 3.0
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

from repetition import (
    compute_statistics,
    find_all_repetitions,
    load_transcription,
)
from voxtral import analyze_audio_segment, load_voxtral_model


def verify_repetitions(
    repetitions: list[dict],
    audio_path: str,
    model,
    processor,
    device: str,
    window: float = 3.0,
) -> list[dict]:
    """Verify repetitions using Voxtral analysis.

    Args:
        repetitions: List of repetition events from find_all_repetitions()
        audio_path: Path to audio file
        model: Loaded Voxtral model
        processor: Loaded processor
        device: Device to run on
        window: Seconds of audio to include before/after the segment

    Returns:
        List of verified events with Voxtral analysis
    """
    verified = []

    for i, rep in enumerate(repetitions):
        print(
            f"  Analyzing repetition {i + 1}/{len(repetitions)}: "
            f'"{rep["phrase"]}" ({rep["n_gram_size"]}-gram) '
            f"by {rep['speaker']} ({rep['count']}x consecutive)"
        )

        start_time = rep["segment_start"] - window
        end_time = rep["segment_end"] + window

        prompt = (
            f"Listen to this audio segment carefully. "
            f'The phrase "{rep["phrase"]}" is repeated {rep["count"]} times consecutively. '
            f"Is this repetition natural (e.g., intentional emphasis, normal speech pattern, "
            f"stuttering, or rhetorical device) or unnatural (e.g., audio glitch, AI looping, "
            f"technical issue, or abnormal speech pattern)?\n\n"
            "Respond in the following format:\n"
            "ANALYSIS: <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the repetition feels natural or problematic>\n"
            "RESULT: <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified.append(
            {
                "type": "repetition",
                "speaker": rep["speaker"],
                "segment_start": rep["segment_start"],
                "segment_end": rep["segment_end"],
                "segment_text": rep["segment_text"],
                "n_gram_size": rep["n_gram_size"],
                "phrase": rep["phrase"],
                "phrase_normalized": rep["phrase_normalized"],
                "count": rep["count"],
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
        description="Verify n-gram repetitions using Voxtral audio analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diarization/verify_repetition.py --transcription output/convo.json --audio audio/convo.wav
  python diarization/verify_repetition.py --transcription output/convo.json --audio audio/convo.wav --min-n 2
  python diarization/verify_repetition.py --transcription output/convo.json --audio audio/convo.wav -o verified.json
        """,
    )
    parser.add_argument(
        "--transcription",
        type=Path,
        required=True,
        help="Path to transcription JSON file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=1,
        help="Minimum n-gram size (default: 1)",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=10,
        help="Maximum n-gram size (default: 10)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="Audio window in seconds to include before/after each segment (default: 3.0)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file for verified results (JSON)",
    )
    args = parser.parse_args()

    # Load transcription data
    print(f"Loading transcription from: {args.transcription}")
    turns = load_transcription(args.transcription)
    print(f"Found {len(turns)} turns")

    speakers = sorted(set(t["speaker"] for t in turns))
    print(f"Speakers: {speakers}")

    # Find repetitions
    print(f"\nFinding repetitions (n-gram size: {args.min_n} to {args.max_n})...")
    repetitions = find_all_repetitions(turns, args.min_n, args.max_n)
    print(f"Found {len(repetitions)} consecutive repetition patterns")

    if not repetitions:
        print("\nNo repetitions found. Nothing to verify.")
        return

    # Compute statistics
    stats = compute_statistics(repetitions, turns)

    # Print summary before verification
    print("\n" + "=" * 50)
    print("REPETITION DETECTION SUMMARY")
    print("=" * 50)
    print(f"Total repetition patterns: {stats['total_repetitions']}")

    if stats["by_speaker"]:
        print("\nRepetitions by speaker:")
        for speaker, count in stats["by_speaker"].items():
            print(f"  {speaker}: {count}")

    if stats["by_ngram_size"]:
        print("\nRepetitions by n-gram size:")
        for n, count in stats["by_ngram_size"].items():
            print(f"  {n}-gram: {count}")

    # Load Voxtral model once
    print("\n" + "-" * 50)
    model, processor, device = load_voxtral_model()

    # Verify repetitions
    print(f"\nVerifying {len(repetitions)} repetitions...")
    verified_repetitions = verify_repetitions(
        repetitions, args.audio, model, processor, device, args.window
    )

    # Print results
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    if verified_repetitions:
        print(f"\n--- REPETITIONS ({len(verified_repetitions)}) ---")
        for event in verified_repetitions:
            print(
                f"\n[{event['segment_start']:.2f}s - {event['segment_end']:.2f}s] "
                f'{event["speaker"]}: "{event["phrase"]}" ({event["count"]}x, {event["n_gram_size"]}-gram)'
            )
            print(f"  Analysis: {event['voxtral_analysis'][:200]}...")

    # Build output
    result = {
        "min_n": args.min_n,
        "max_n": args.max_n,
        "window": args.window,
        "summary": {
            "total_repetitions": len(verified_repetitions),
            "by_speaker": stats["by_speaker"],
            "by_ngram_size": stats["by_ngram_size"],
        },
        "repetitions": verified_repetitions,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
