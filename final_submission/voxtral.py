"""Voxtral audio analysis utilities.

Requires: pip install "transformers[torch]==v4.56.1"
"""

import os
import re
from typing import NamedTuple


class VoxtralParsedResponse(NamedTuple):
    """Parsed response from Voxtral model."""

    analysis: str
    result: str  # "natural" or "unnatural"


def parse_voxtral_response(response: str) -> VoxtralParsedResponse:
    """Extract Analysis and Result sections from Voxtral response.

    Args:
        response: Raw response text from Voxtral model.

    Returns:
        VoxtralParsedResponse with analysis and result fields.
    """
    # Extract analysis section
    analysis_match = re.search(
        r"\*\*ANALYSIS:\*\*\s*(.*?)(?=\*\*RESULT|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    analysis = analysis_match.group(1).strip() if analysis_match else ""

    # Extract result section (natural/unnatural)
    result_match = re.search(r"\*\*RESULT:?\*\*\s*(\w+)", response, re.IGNORECASE)
    result = result_match.group(1).strip().lower() if result_match else ""

    return VoxtralParsedResponse(analysis=analysis, result=result)


def compute_latency_metrics(verified_results: dict) -> dict:
    """Compute latency metrics from verified results.

    Args:
        verified_results: Dict containing slow_responses and intra_speaker_pauses
                          with voxtral_analysis fields.

    Returns:
        Dict with average_latency and unnatural_pauses count.
    """
    all_events = verified_results.get("slow_responses", []) + verified_results.get(
        "intra_speaker_pauses", []
    )

    if not all_events:
        return {"average_latency": 0.0, "unnatural_pauses": 0}

    # Calculate average latency
    latencies = []
    unnatural_count = 0

    for event in all_events:
        # Get latency (slow_responses use "latency", intra_speaker_pauses use "duration")
        latency = event.get("latency") or event.get("duration", 0)
        latencies.append(latency)

        # Parse voxtral analysis to count unnatural pauses
        voxtral_analysis = event.get("voxtral_analysis", "")
        if voxtral_analysis:
            parsed = parse_voxtral_response(voxtral_analysis)
            if parsed.result == "unnatural":
                unnatural_count += 1

    average_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return {
        "average_latency": round(average_latency, 3),
        "unnatural_pauses": unnatural_count,
    }


REPO_ID = "mistralai/Voxtral-Mini-3B-2507"
SAMPLE_RATE = 16000


def load_voxtral_model(device: str = None):
    """Load Voxtral model and processor once for reuse.

    Args:
        device: Device to load model on. Defaults to "mps" if available, else "cpu".

    Returns:
        Tuple of (model, processor, device)
    """
    import torch
    from transformers import AutoProcessor, VoxtralForConditionalGeneration

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading Voxtral model on {device}...")
    processor = AutoProcessor.from_pretrained(REPO_ID)
    model = VoxtralForConditionalGeneration.from_pretrained(
        REPO_ID, torch_dtype=torch.bfloat16, device_map=device
    )

    return model, processor, device


def analyze_audio_segment(
    audio_path: str,
    prompt: str,
    start_time: float,
    end_time: float,
    model,
    processor,
    device: str,
    max_new_tokens: int = 500,
) -> str:
    """Analyze an audio segment using a pre-loaded Voxtral model.

    Args:
        audio_path: Path to audio file
        prompt: Prompt for the model
        start_time: Start time in seconds
        end_time: End time in seconds
        model: Pre-loaded Voxtral model
        processor: Pre-loaded processor
        device: Device model is on
        max_new_tokens: Max tokens to generate

    Returns:
        Model's text response
    """
    import soundfile as sf
    import torch
    from transformers.audio_utils import load_audio

    audio = load_audio(audio_path, sampling_rate=SAMPLE_RATE)
    audio_duration = audio.shape[0] / SAMPLE_RATE

    # Clamp to valid bounds
    start_time = max(0, start_time) if start_time is not None else 0
    end_time = min(audio_duration, end_time) if end_time is not None else audio_duration

    start_sample = int(start_time * SAMPLE_RATE)
    end_sample = int(end_time * SAMPLE_RATE)
    audio_segment = audio[start_sample:end_sample]

    # Save to temp file for processing
    temp_audio_path = f"temp_voxtral_{start_time:.1f}_{end_time:.1f}.wav"
    sf.write(temp_audio_path, audio_segment, samplerate=SAMPLE_RATE)

    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": temp_audio_path},
                ],
            }
        ]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(device, dtype=torch.bfloat16)

        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return decoded[0] if decoded else ""
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def understand_audio(audio_path, prompt, start_time=None, end_time=None):
    """Analyze audio with Voxtral (loads model each call - use analyze_audio_segment for batch).

    This is a convenience function that loads the model each time.
    For analyzing multiple segments, use load_voxtral_model() once,
    then call analyze_audio_segment() for each segment.
    """
    import soundfile as sf
    import torch
    from transformers.audio_utils import load_audio

    audio = load_audio(audio_path, sampling_rate=SAMPLE_RATE)
    print(f"Audio duration (seconds): {audio.shape[0] / SAMPLE_RATE}")

    start_sample = int(start_time * SAMPLE_RATE) if start_time is not None else 0
    end_sample = int(end_time * SAMPLE_RATE) if end_time is not None else audio.shape[0]
    audio = audio[start_sample:end_sample]
    print(f"Extracted audio duration (seconds): {audio.shape[0] / SAMPLE_RATE}")

    model, processor, device = load_voxtral_model()

    temp_audio_path = "temp_audio.wav"
    sf.write(temp_audio_path, audio, samplerate=SAMPLE_RATE)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": temp_audio_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(conversation)
    inputs = inputs.to(device, dtype=torch.bfloat16)

    outputs = model.generate(**inputs, max_new_tokens=500)
    outputs = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )

    os.remove(temp_audio_path)

    return outputs


def compute_latency_metrics_from_file(results_path: str) -> dict:
    """Load a verified results JSON file and compute latency metrics.

    Args:
        results_path: Path to the verified_results.json file.

    Returns:
        Dict with average_latency and unnatural_pauses count.
    """
    import json

    with open(results_path, "r") as f:
        verified_results = json.load(f)

    return compute_latency_metrics(verified_results)
