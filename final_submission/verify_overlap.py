"""Verify speaker overlaps using Voxtral audio analysis."""

from overlap import (
    add_interruption_info,
    build_annotation,
    find_overlaps,
    load_diarization,
    load_diarization_from_dict,
    merge_adjacent_overlaps,
)
from voxtral import analyze_audio_segment, load_voxtral_model, parse_voxtral_response


def extract_results(event: dict) -> dict:
    """Extract Analysis and Result fields from a verified event.

    Args:
        event: A verified event dict containing 'voxtral_analysis' field.

    Returns:
        Dict with 'analysis' and 'result' fields extracted from voxtral_analysis.
    """
    voxtral_analysis = event.get("voxtral_analysis", "")
    if not voxtral_analysis:
        return {"analysis": "", "result": ""}

    parsed = parse_voxtral_response(voxtral_analysis)
    return {"analysis": parsed.analysis, "result": parsed.result}


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
            "**ANALYSIS:** <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the overlap feels natural or problematic>\n"
            "**RESULT:** <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified_event = {
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
        verified_event["extracted"] = extract_results(verified_event)
        verified.append(verified_event)

    return verified


def run_overlap_verification(
    diarization_data: dict,
    audio_path: str,
    threshold: float = 0.5,
    window: float = 5.0,
    merge: bool = False,
    model=None,
    processor=None,
    device: str = None,
) -> dict:
    """Run complete overlap verification pipeline.

    Args:
        diarization_data: Diarization output dict (from pyannote API or file)
        audio_path: Path to audio file
        threshold: Minimum overlap duration in seconds to analyze
        window: Audio window in seconds to include before/after each overlap
        merge: Whether to merge adjacent overlaps with the same speakers
        model: Pre-loaded Voxtral model (optional, will load if not provided)
        processor: Pre-loaded processor (optional)
        device: Device to run on (optional)

    Returns:
        Dict with summary and verified events
    """
    # Load diarization segments
    segments = load_diarization_from_dict(diarization_data)
    print(f"Found {len(segments)} segments")

    # Build annotation
    annotation = build_annotation(segments)
    speakers = annotation.labels()
    print(f"Speakers: {speakers}")

    # Find overlaps
    overlaps = find_overlaps(annotation)

    if merge:
        overlaps = merge_adjacent_overlaps(overlaps)
        print(f"Found {len(overlaps)} overlap regions (merged)")
    else:
        print(f"Found {len(overlaps)} overlap regions")

    # Add interruption classification
    overlaps = add_interruption_info(overlaps, annotation)

    # Filter by threshold
    overlaps = [o for o in overlaps if o["duration"] >= threshold]
    print(f"Overlaps above threshold ({threshold}s): {len(overlaps)}")

    if not overlaps:
        print(f"No overlaps above threshold ({threshold}s) found.")
        return {
            "threshold": threshold,
            "window": window,
            "summary": {
                "total_overlaps": 0,
                "natural_count": 0,
                "unnatural_count": 0,
            },
            "overlaps": [],
        }

    # Load Voxtral model if not provided
    if model is None:
        model, processor, device = load_voxtral_model()

    # Verify overlaps
    print(f"Verifying {len(overlaps)} overlaps...")
    verified_overlaps = verify_overlaps(
        overlaps, audio_path, model, processor, device, window
    )

    # Collect all speakers involved in overlaps
    all_speakers = set()
    for e in verified_overlaps:
        for speaker in e.get("speakers", []):
            all_speakers.add(speaker)

    # Count natural vs unnatural by speaker
    natural_count = {speaker: 0 for speaker in all_speakers}
    unnatural_count = {speaker: 0 for speaker in all_speakers}

    for e in verified_overlaps:
        result = e.get("extracted", {}).get("result")
        for speaker in e.get("speakers", []):
            if result == "natural":
                natural_count[speaker] += 1
            elif result == "unnatural":
                unnatural_count[speaker] += 1

    return {
        "threshold": threshold,
        "window": window,
        "summary": {
            "total_overlaps": len(verified_overlaps),
            "natural_count": natural_count,
            "unnatural_count": unnatural_count,
        },
        "overlaps": verified_overlaps,
    }
