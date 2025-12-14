"""Verify slow responses and intra-speaker pauses using Voxtral audio analysis."""

from latency import (
    build_annotation,
    filter_clean_transitions,
    find_intra_speaker_pauses,
    find_turn_transitions,
    flag_slow_responses,
    load_diarization,
    load_diarization_from_dict,
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
            "**ANALYSIS:** <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the pause feels natural or awkward>\n"
            "**RESULT:** <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified_event = {
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
        verified_event["extracted"] = extract_results(verified_event)
        verified.append(verified_event)

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
            "**ANALYSIS:** <your free-form analysis here - explain your reasoning, consider the context, "
            "tone, and whether the pause feels natural or awkward>\n"
            "**RESULT:** <either 'natural' or 'unnatural'>"
        )

        analysis = analyze_audio_segment(
            audio_path, prompt, start_time, end_time, model, processor, device
        )

        verified_event = {
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
        verified_event["extracted"] = extract_results(verified_event)
        verified.append(verified_event)

    return verified


def run_latency_verification(
    diarization_data: dict,
    audio_path: str,
    threshold: float = 2.0,
    window: float = 5.0,
    model=None,
    processor=None,
    device: str = None,
) -> dict:
    """Run complete latency verification pipeline.

    Args:
        diarization_data: Diarization output dict (from pyannote API or file)
        audio_path: Path to audio file
        threshold: Latency threshold in seconds for flagging
        window: Audio window in seconds to include before/after each gap
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

    # Find slow responses (inter-speaker)
    transitions = find_turn_transitions(annotation)
    clean_transitions = filter_clean_transitions(transitions, annotation)
    slow_responses = flag_slow_responses(clean_transitions, threshold=threshold)
    print(f"Found {len(slow_responses)} slow responses (>= {threshold}s)")

    # Find intra-speaker pauses
    intra_pauses = find_intra_speaker_pauses(annotation, threshold=threshold)
    print(f"Found {len(intra_pauses)} intra-speaker pauses (>= {threshold}s)")

    total_events = len(slow_responses) + len(intra_pauses)

    if total_events == 0:
        print(f"No events above threshold ({threshold}s) found.")
        return {
            "threshold": threshold,
            "window": window,
            "summary": {
                "total_events": 0,
                "slow_responses": 0,
                "intra_speaker_pauses": 0,
                "unnatural_count": 0,
                "natural_count": 0,
            },
            "slow_responses": [],
            "intra_speaker_pauses": [],
        }

    # Load Voxtral model if not provided
    if model is None:
        model, processor, device = load_voxtral_model()

    # Verify slow responses
    verified_slow_responses = []
    if slow_responses:
        print(f"Verifying {len(slow_responses)} slow responses...")
        verified_slow_responses = verify_slow_responses(
            slow_responses, audio_path, model, processor, device, window
        )

    # Verify intra-speaker pauses
    verified_intra_pauses = []
    if intra_pauses:
        print(f"Verifying {len(intra_pauses)} intra-speaker pauses...")
        verified_intra_pauses = verify_intra_speaker_pauses(
            intra_pauses, audio_path, model, processor, device, window
        )

    # Collect all speakers
    all_events = verified_slow_responses + verified_intra_pauses
    all_speakers = set()
    for e in verified_slow_responses:
        if e.get("prev_speaker"):
            all_speakers.add(e["prev_speaker"])
    for e in verified_intra_pauses:
        if e.get("speaker"):
            all_speakers.add(e["speaker"])

    # Count natural vs unnatural by speaker
    natural_count = {speaker: 0 for speaker in all_speakers}
    unnatural_count = {speaker: 0 for speaker in all_speakers}

    for e in verified_slow_responses:
        speaker = e.get("prev_speaker")
        result = e.get("extracted", {}).get("result")
        if speaker and result == "natural":
            natural_count[speaker] += 1
        elif speaker and result == "unnatural":
            unnatural_count[speaker] += 1

    for e in verified_intra_pauses:
        speaker = e.get("speaker")
        result = e.get("extracted", {}).get("result")
        if speaker and result == "natural":
            natural_count[speaker] += 1
        elif speaker and result == "unnatural":
            unnatural_count[speaker] += 1

    # Calculate average and max latency by speaker
    latencies_by_speaker = {speaker: [] for speaker in all_speakers}

    for e in verified_slow_responses:
        speaker = e.get("prev_speaker")
        if speaker:
            latencies_by_speaker[speaker].append(e.get("latency", 0))

    for e in verified_intra_pauses:
        speaker = e.get("speaker")
        if speaker:
            latencies_by_speaker[speaker].append(e.get("duration", 0))

    average_latency = {}
    max_latency = {}
    for speaker, latencies in latencies_by_speaker.items():
        if latencies:
            average_latency[speaker] = round(sum(latencies) / len(latencies), 3)
            max_latency[speaker] = round(max(latencies), 3)
        else:
            average_latency[speaker] = 0
            max_latency[speaker] = 0

    return {
        "threshold": threshold,
        "window": window,
        "summary": {
            "total_events": total_events,
            "slow_responses": len(verified_slow_responses),
            "intra_speaker_pauses": len(verified_intra_pauses),
            "natural_count": natural_count,
            "unnatural_count": unnatural_count,
            "average_latency": average_latency,
            "max_latency": max_latency,
        },
        "slow_responses": verified_slow_responses,
        "intra_speaker_pauses": verified_intra_pauses,
    }
