import argparse
import json
import time
from pathlib import Path

import requests

PYANNOTE_API_KEY = "sk_6faa8aab870f4a37ab158ad1b735c053"


def main():
    parser = argparse.ArgumentParser(
        description="Diarize an audio file using pyannoteAI API"
    )
    parser.add_argument(
        "url",
        nargs="?",
        default="https://files.pyannote.ai/marklex1min.wav",
        help="Public URL of the audio file (default: sample file)",
    )
    args = parser.parse_args()

    if not PYANNOTE_API_KEY:
        raise ValueError("Please set PYANNOTE_API_KEY environment variable")

    audio_url = args.url

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
        print(f"Error: {response.status_code} - {response.text}")
        exit(1)

    job_data = response.json()
    job_id = job_data["jobId"]
    print(f"Job created: {job_id}")
    print(f"Status: {job_data['status']}")

    # Step 2: Poll for results
    print("\nPolling for results...")
    headers = {"Authorization": f"Bearer {PYANNOTE_API_KEY}"}

    while True:
        response = requests.get(
            f"https://api.pyannote.ai/v1/jobs/{job_id}", headers=headers
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            break

        result = response.json()
        status = result["status"]

        if status in ["succeeded", "failed", "canceled"]:
            if status == "succeeded":
                print("\n✅ Job completed successfully!")
                print("\nDiarization output:")
                print(result["output"])

                # Write result to output folder
                output_dir = Path(__file__).parent / "output"
                output_dir.mkdir(exist_ok=True)
                output_file = output_dir / f"{job_id}.json"
                with open(output_file, "w") as f:
                    json.dump(result["output"], f, indent=2)
                print(f"\n📁 Result saved to: {output_file}")
            else:
                print(f"\n❌ Job {status}")
                print(result)
            break

        print(f"Status: {status}, waiting 10 seconds...")
        time.sleep(10)


if __name__ == "__main__":
    main()
