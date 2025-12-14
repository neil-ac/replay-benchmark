"""Verify slow responses and intra-speaker pauses using Voxtral audio analysis.

This script detects gaps above a threshold (both inter-speaker and intra-speaker)
and uses Voxtral to analyze if they represent missed turns/omissions.

Example:
    python diarization/verify_latency.py \\
        --diarization output/convo_diarization.json \\
        --audio audio/convo.wav \\
        --threshold 2.0 \\
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

from latency import (
    build_annotation,
    filter_clean_transitions,
    find_intra_speaker_pauses,
    find_turn_transitions,
    flag_slow_responses,
    load_diarization,
)
from voxtral import analyze_audio_segment, load_voxtral_model


def verify_slow_responses(
    slow_responses: list[dict],
    audio_path: str,
    model,
    processor,
    device: str,
    window: float = 5.0,
) -> list[dict]:
    """Verify slow responses using Voxtral analysis.

    Args:
        slow_responses: List of slow response events from flag_slow_responses()
        audio_path: Path to audio file
        model: Loaded Voxtral model
        processor: Loaded processor
        device: Device to run on
        window: Seconds of audio to include before/after the gap

    Returns:
        List of verified events with Voxtral analysis
    """
    verified = []

    for i, event in enumerate(slow_responses):
        print(
            f"  Analyzing slow response {i + 1}/{len(slow_responses)}: "
            f"{event['prev_speaker']} -> {event['next_speaker']} "
            f"({event['latency']:.2f}s gap at {event['prev_end']:.1f}s)"
        )

        start_time = event["prev_end"] - window
        end_time = event["next_start"] + window

        prompt = (
            "Listen to this audio segment carefully. There is a pause/gap in the conversation. "
            "One speaker finishes talking, then there is silence before the next speaker responds. "
            "Was the second speaker supposed to respond sooner? Did they miss their turn or fail to respond? "
            "Is this an awkward pause where someone should have spoken?\n\n"
            "Respond in the following format:\n"
            "ANALYSIS: <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the pause feels natural or awkward>\n"
            "RESULT: <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified.append(
            {
                "type": "slow_response",
                "prev_speaker": event["prev_speaker"],
                "next_speaker": event["next_speaker"],
                "gap_start": event["prev_end"],
                "gap_end": event["next_start"],
                "latency": event["latency"],
                "audio_window": {
                    "start": round(max(0, start_time), 3),
                    "end": round(end_time, 3),
                },
                "voxtral_analysis": analysis,
            }
        )

    return verified


def verify_intra_speaker_pauses(
    pauses: list[dict],
    audio_path: str,
    model,
    processor,
    device: str,
    window: float = 5.0,
) -> list[dict]:
    """Verify intra-speaker pauses using Voxtral analysis.

    Args:
        pauses: List of intra-speaker pause events from find_intra_speaker_pauses()
        audio_path: Path to audio file
        model: Loaded Voxtral model
        processor: Loaded processor
        device: Device to run on
        window: Seconds of audio to include before/after the gap

    Returns:
        List of verified events with Voxtral analysis
    """
    verified = []

    for i, pause in enumerate(pauses):
        print(
            f"  Analyzing intra-speaker pause {i + 1}/{len(pauses)}: "
            f"{pause['speaker']} ({pause['duration']:.2f}s gap at {pause['start']:.1f}s)"
        )

        start_time = pause["start"] - window
        end_time = pause["end"] + window

        prompt = (
            "Listen to this audio segment carefully. There is a pause in the middle of a conversation. "
            "One person is speaking, then there is silence, then they continue speaking again. "
            "During this pause, was the other person supposed to respond but failed to? "
            "Did the speaker pause to wait for a response that never came?\n\n"
            "Respond in the following format:\n"
            "ANALYSIS: <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the pause feels natural or awkward>\n"
            "RESULT: <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified.append(
            {
                "type": "intra_speaker_pause",
                "speaker": pause["speaker"],
                "gap_start": pause["start"],
                "gap_end": pause["end"],
                "duration": pause["duration"],
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
        description="Verify slow responses and intra-speaker pauses using Voxtral audio analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python diarization/verify_latency.py --diarization output/convo.json --audio audio/convo.wav
  python diarization/verify_latency.py --diarization output/convo.json --audio audio/convo.wav --threshold 3.0
  python diarization/verify_latency.py --diarization output/convo.json --audio audio/convo.wav -o verified.json
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
        default=2.0,
        help="Latency threshold in seconds for flagging (default: 2.0)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Audio window in seconds to include before/after each gap (default: 5.0)",
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

    # Find slow responses (inter-speaker)
    transitions = find_turn_transitions(annotation)
    clean_transitions = filter_clean_transitions(transitions, annotation)
    slow_responses = flag_slow_responses(clean_transitions, threshold=args.threshold)
    print(f"Found {len(slow_responses)} slow responses (>= {args.threshold}s)")

    # Find intra-speaker pauses
    intra_pauses = find_intra_speaker_pauses(annotation, threshold=args.threshold)
    print(f"Found {len(intra_pauses)} intra-speaker pauses (>= {args.threshold}s)")

    total_events = len(slow_responses) + len(intra_pauses)

    if total_events == 0:
        print(
            f"\nNo events above threshold ({args.threshold}s) found. Nothing to verify."
        )
        return

    # Load Voxtral model once
    model, processor, device = load_voxtral_model()

    # Verify slow responses
    verified_slow_responses = []
    if slow_responses:
        print(f"\nVerifying {len(slow_responses)} slow responses...")
        verified_slow_responses = verify_slow_responses(
            slow_responses, args.audio, model, processor, device, args.window
        )

    # Verify intra-speaker pauses
    verified_intra_pauses = []
    if intra_pauses:
        print(f"\nVerifying {len(intra_pauses)} intra-speaker pauses...")
        verified_intra_pauses = verify_intra_speaker_pauses(
            intra_pauses, args.audio, model, processor, device, args.window
        )

    # Print results
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    if verified_slow_responses:
        print(f"\n--- SLOW RESPONSES ({len(verified_slow_responses)}) ---")
        for event in verified_slow_responses:
            print(
                f"\n[{event['gap_start']:.2f}s - {event['gap_end']:.2f}s] "
                f"{event['prev_speaker']} -> {event['next_speaker']} "
                f"({event['latency']:.2f}s gap)"
            )
            print(f"  Analysis: {event['voxtral_analysis'][:200]}...")

    if verified_intra_pauses:
        print(f"\n--- INTRA-SPEAKER PAUSES ({len(verified_intra_pauses)}) ---")
        for pause in verified_intra_pauses:
            print(
                f"\n[{pause['gap_start']:.2f}s - {pause['gap_end']:.2f}s] "
                f"{pause['speaker']} ({pause['duration']:.2f}s gap)"
            )
            print(f"  Analysis: {pause['voxtral_analysis'][:200]}...")

    # Build output
    result = {
        "threshold": args.threshold,
        "window": args.window,
        "summary": {
            "total_events": total_events,
            "slow_responses": len(verified_slow_responses),
            "intra_speaker_pauses": len(verified_intra_pauses),
        },
        "slow_responses": verified_slow_responses,
        "intra_speaker_pauses": verified_intra_pauses,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
