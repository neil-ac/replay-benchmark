import argparse
import uuid
from pathlib import Path
import os
import requests
from dotenv import load_dotenv


PYANNOTE_API_KEY = os.getenv("PYANNOTE_API_KEY")

load_dotenv()


def upload_audio_file(input_path: str, object_key: str | None = None) -> str:
    """
    Upload a local audio file to pyannote's temporary storage.

    Args:
        input_path: Path to the local audio file
        object_key: Optional custom object key. If not provided, a UUID will be generated.

    Returns:
        The media URL (media://{object_key}) to use in subsequent API calls.

    Note:
        Files are temporarily stored and automatically removed within 48 hours.
    """
    if not PYANNOTE_API_KEY:
        raise ValueError("Please set PYANNOTE_API_KEY")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    # Generate object key if not provided
    if object_key is None:
        object_key = str(uuid.uuid4())

    media_url = f"media://{object_key}"

    # Step 1: Create a pre-signed PUT URL
    print(f"Creating temporary storage location: {media_url}")
    headers = {
        "Authorization": f"Bearer {PYANNOTE_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://api.pyannote.ai/v1/media/input",
        json={"url": media_url},
        headers=headers,
    )

    if response.status_code not in [200, 201]:
        print(
            f"Error creating storage location: {response.status_code} - {response.text}"
        )
        raise RuntimeError(f"Failed to create storage location: {response.text}")

    data = response.json()
    presigned_url = data["url"]

    # Step 2: Upload local file to the pre-signed URL
    print(f"Uploading {input_path} to temporary storage...")

    with open(input_path, "rb") as f:
        upload_response = requests.put(
            presigned_url,
            data=f,
            headers={"Content-Type": "application/octet-stream"},
        )

    if upload_response.status_code not in [200, 201]:
        print(
            f"Error uploading file: {upload_response.status_code} - {upload_response.text}"
        )
        raise RuntimeError(f"Failed to upload file: {upload_response.text}")

    print(f"✅ File uploaded successfully!")
    print(f"📁 Use this URL in API calls: {media_url}")

    return media_url


def main():
    parser = argparse.ArgumentParser(
        description="Upload a local audio file to pyannote's temporary storage"
    )
    parser.add_argument(
        "input_path",
        help="Path to the local audio file to upload",
    )
    parser.add_argument(
        "--key",
        "-k",
        dest="object_key",
        default=None,
        help="Custom object key for the uploaded file (default: auto-generated UUID)",
    )
    args = parser.parse_args()

    media_url = upload_audio_file(args.input_path, args.object_key)
    print(f"\nTo diarize this file, run:")
    print(f"  uv run diarization/diarize.py '{media_url}'")


if __name__ == "__main__":
    main()
