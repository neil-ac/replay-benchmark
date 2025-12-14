"""Diarize audio files using pyannoteAI API."""

import json
import time
from pathlib import Path
import os
import requests
from dotenv import load_dotenv

load_dotenv()

PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")


def run_diarization(audio_url: str) -> dict:
    """Run diarization on an audio file using pyannoteAI API.

    Args:
        audio_url: URL of the audio file (either public URL or media:// URL from upload)

    Returns:
        Dict containing diarization and transcription output from pyannote API
    """
    if not PYANNOTE_API_KEY:
        raise ValueError("Please set PYANNOTE_API_KEY environment variable")

    # Step 1: Submit diarization job
    print(f"Submitting diarization job for: {audio_url}")
    headers = {
        "Authorization": f"Bearer {PYANNOTE_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {"url": audio_url, "transcription": True, "exclusive": True}

    response = requests.post(
        "https://api.pyannote.ai/v1/diarize", headers=headers, json=data
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to submit job: {response.status_code} - {response.text}"
        )

    job_data = response.json()
    job_id = job_data["jobId"]
    print(f"Job created: {job_id}")
    print(f"Status: {job_data['status']}")

    # Step 2: Poll for results
    print("Polling for results...")
    headers = {"Authorization": f"Bearer {PYANNOTE_API_KEY}"}

    while True:
        response = requests.get(
            f"https://api.pyannote.ai/v1/jobs/{job_id}", headers=headers
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get job status: {response.status_code} - {response.text}"
            )

        result = response.json()
        status = result["status"]

        if status in ["succeeded", "failed", "canceled"]:
            if status == "succeeded":
                print("Job completed successfully!")
                return result["output"]
            else:
                raise RuntimeError(f"Diarization job {status}")

        print(f"Status: {status}, waiting 10 seconds...")
        time.sleep(10)
