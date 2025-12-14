"""Verify n-gram repetitions using Voxtral audio analysis."""

from repetition import (
    compute_statistics,
    find_all_repetitions,
    load_transcription,
    load_transcription_from_dict,
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
            "**ANALYSIS:** <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the repetition feels natural or problematic>\n"
            "**RESULT:** <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified_event = {
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
        verified_event["extracted"] = extract_results(verified_event)
        verified.append(verified_event)

    return verified


def run_repetition_verification(
    transcription_data: dict,
    audio_path: str,
    min_n: int = 1,
    max_n: int = 10,
    window: float = 3.0,
    model=None,
    processor=None,
    device: str = None,
) -> dict:
    """Run complete repetition verification pipeline.

    Args:
        transcription_data: Transcription output dict (from pyannote API or file)
        audio_path: Path to audio file
        min_n: Minimum n-gram size
        max_n: Maximum n-gram size
        window: Audio window in seconds to include before/after each segment
        model: Pre-loaded Voxtral model (optional, will load if not provided)
        processor: Pre-loaded processor (optional)
        device: Device to run on (optional)

    Returns:
        Dict with summary and verified events
    """
    # Load transcription turns
    turns = load_transcription_from_dict(transcription_data)
    print(f"Found {len(turns)} turns")

    speakers = sorted(set(t["speaker"] for t in turns))
    print(f"Speakers: {speakers}")

    # Find repetitions
    print(f"Finding repetitions (n-gram size: {min_n} to {max_n})...")
    repetitions = find_all_repetitions(turns, min_n, max_n)
    print(f"Found {len(repetitions)} consecutive repetition patterns")

    if not repetitions:
        print("No repetitions found.")
        return {
            "min_n": min_n,
            "max_n": max_n,
            "window": window,
            "summary": {
                "total_repetitions": 0,
                "by_speaker": {},
                "by_ngram_size": {},
                "natural_count": 0,
                "unnatural_count": 0,
            },
            "repetitions": [],
        }

    # Compute statistics
    stats = compute_statistics(repetitions, turns)

    # Load Voxtral model if not provided
    if model is None:
        model, processor, device = load_voxtral_model()

    # Verify repetitions
    print(f"Verifying {len(repetitions)} repetitions...")
    verified_repetitions = verify_repetitions(
        repetitions, audio_path, model, processor, device, window
    )

    # Collect all speakers
    all_speakers = set(
        e.get("speaker") for e in verified_repetitions if e.get("speaker")
    )

    # Count natural vs unnatural by speaker
    natural_count = {speaker: 0 for speaker in all_speakers}
    unnatural_count = {speaker: 0 for speaker in all_speakers}

    for e in verified_repetitions:
        speaker = e.get("speaker")
        result = e.get("extracted", {}).get("result")
        if speaker and result == "natural":
            natural_count[speaker] += 1
        elif speaker and result == "unnatural":
            unnatural_count[speaker] += 1

    return {
        "min_n": min_n,
        "max_n": max_n,
        "window": window,
        "summary": {
            "total_repetitions": len(verified_repetitions),
            "by_ngram_size": stats["by_ngram_size"],
            "natural_count": natural_count,
            "unnatural_count": unnatural_count,
        },
        "repetitions": verified_repetitions,
    }
