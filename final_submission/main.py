"""Main entrypoint for S2S evaluation pipeline.

This script processes a .wav audio file and outputs consolidated metrics
from latency, overlap, and repetition verification analyses.

Usage:
    python main.py <audio_path> [--output <output_path>]

Example:
    python final_submission/main.py audio/convo.wav --output results.json
"""

import argparse
import json
from pathlib import Path

from diarize import run_diarization
from upload import upload_audio_file
from verify_latency import run_latency_verification
from verify_overlap import run_overlap_verification
from verify_repetition import run_repetition_verification
from voxtral import load_voxtral_model


def run_pipeline(
    audio_path: str,
    output_path: str = None,
    latency_threshold: float = 2.0,
    overlap_threshold: float = 0.5,
    latency_window: float = 5.0,
    overlap_window: float = 5.0,
    repetition_window: float = 3.0,
    min_n: int = 1,
    max_n: int = 10,
) -> dict:
    """Run the complete S2S evaluation pipeline.

    Args:
        audio_path: Path to .wav audio file
        output_path: Optional path to save JSON output
        latency_threshold: Threshold in seconds for flagging slow responses
        overlap_threshold: Minimum overlap duration in seconds to analyze
        latency_window: Audio window for latency verification
        overlap_window: Audio window for overlap verification
        repetition_window: Audio window for repetition verification
        min_n: Minimum n-gram size for repetition detection
        max_n: Maximum n-gram size for repetition detection

    Returns:
        Dict with consolidated metrics from all verification modules
    """
    audio_path = str(Path(audio_path).resolve())
    print(f"Processing audio file: {audio_path}")

    # Step 1: Upload audio to pyannote temporary storage
    print("\n" + "=" * 60)
    print("STEP 1: Uploading audio to pyannote")
    print("=" * 60)
    media_url = upload_audio_file(audio_path)

    # Step 2: Run diarization
    print("\n" + "=" * 60)
    print("STEP 2: Running diarization")
    print("=" * 60)
    diarization_output = run_diarization(media_url)
    print(
        f"Diarization complete. Found {len(diarization_output.get('diarization', []))} segments"
    )

    # Step 3: Load Voxtral model once for all verifications
    print("\n" + "=" * 60)
    print("STEP 3: Loading Voxtral model")
    print("=" * 60)
    model, processor, device = load_voxtral_model()

    # Step 4: Run latency verification
    print("\n" + "=" * 60)
    print("STEP 4: Running latency verification")
    print("=" * 60)
    latency_results = run_latency_verification(
        diarization_data=diarization_output,
        audio_path=audio_path,
        threshold=latency_threshold,
        window=latency_window,
        model=model,
        processor=processor,
        device=device,
    )

    # Step 5: Run overlap verification
    print("\n" + "=" * 60)
    print("STEP 5: Running overlap verification")
    print("=" * 60)
    overlap_results = run_overlap_verification(
        diarization_data=diarization_output,
        audio_path=audio_path,
        threshold=overlap_threshold,
        window=overlap_window,
        model=model,
        processor=processor,
        device=device,
    )

    # Step 6: Run repetition verification
    print("\n" + "=" * 60)
    print("STEP 6: Running repetition verification")
    print("=" * 60)
    repetition_results = run_repetition_verification(
        transcription_data=diarization_output,
        audio_path=audio_path,
        min_n=min_n,
        max_n=max_n,
        window=repetition_window,
        model=model,
        processor=processor,
        device=device,
    )

    # Step 7: Consolidate results
    print("\n" + "=" * 60)
    print("STEP 7: Consolidating results")
    print("=" * 60)

    consolidated_results = {
        "audio_file": audio_path,
        "latency": {
            "summary": latency_results["summary"],
            "events": latency_results["slow_responses"]
            + latency_results["intra_speaker_pauses"],
        },
        "overlap": {
            "summary": overlap_results["summary"],
            "events": overlap_results["overlaps"],
        },
        "repetition": {
            "summary": repetition_results["summary"],
            "events": repetition_results["repetitions"],
        },
    }

    # Print summary
    print("\n" + "=" * 60)
    print("CONSOLIDATED METRICS SUMMARY")
    print("=" * 60)

    print("\nLATENCY:")
    print(f"  Total events: {latency_results['summary']['total_events']}")
    print(f"  Slow responses: {latency_results['summary']['slow_responses']}")
    print(
        f"  Intra-speaker pauses: {latency_results['summary']['intra_speaker_pauses']}"
    )
    print(f"  Natural (by speaker): {latency_results['summary']['natural_count']}")
    print(f"  Unnatural (by speaker): {latency_results['summary']['unnatural_count']}")
    print(
        f"  Average latency (by speaker): {latency_results['summary']['average_latency']}"
    )
    print(f"  Max latency (by speaker): {latency_results['summary']['max_latency']}")

    print("\nOVERLAP:")
    print(f"  Total overlaps: {overlap_results['summary']['total_overlaps']}")
    print(f"  Natural (by speaker): {overlap_results['summary']['natural_count']}")
    print(f"  Unnatural (by speaker): {overlap_results['summary']['unnatural_count']}")

    print("\nREPETITION:")
    print(f"  Total repetitions: {repetition_results['summary']['total_repetitions']}")
    print(f"  Natural (by speaker): {repetition_results['summary']['natural_count']}")
    print(
        f"  Unnatural (by speaker): {repetition_results['summary']['unnatural_count']}"
    )

    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(consolidated_results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return consolidated_results


def main():
    parser = argparse.ArgumentParser(
        description="Run S2S evaluation pipeline on audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py audio/conversation.wav
    python main.py audio/conversation.wav --output results.json
    python main.py audio/conversation.wav --latency-threshold 3.0 --overlap-threshold 0.3
        """,
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Path to .wav audio file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output path for JSON results (default: evaluation_results.json)",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=2.0,
        help="Latency threshold in seconds for flagging (default: 2.0)",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.2,
        help="Minimum overlap duration in seconds to analyze (default: 0.5)",
    )
    parser.add_argument(
        "--latency-window",
        type=float,
        default=5.0,
        help="Audio window for latency analysis (default: 5.0)",
    )
    parser.add_argument(
        "--overlap-window",
        type=float,
        default=5.0,
        help="Audio window for overlap analysis (default: 5.0)",
    )
    parser.add_argument(
        "--repetition-window",
        type=float,
        default=3.0,
        help="Audio window for repetition analysis (default: 3.0)",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=1,
        help="Minimum n-gram size for repetition detection (default: 1)",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=10,
        help="Maximum n-gram size for repetition detection (default: 10)",
    )

    args = parser.parse_args()

    run_pipeline(
        audio_path=args.audio,
        output_path=args.output,
        latency_threshold=args.latency_threshold,
        overlap_threshold=args.overlap_threshold,
        latency_window=args.latency_window,
        overlap_window=args.overlap_window,
        repetition_window=args.repetition_window,
        min_n=args.min_n,
        max_n=args.max_n,
    )


if __name__ == "__main__":
    main()
