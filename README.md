# S2S Evals

Evaluating speech-to-speech (S2S) models.

## Diarization

Scripts for speaker diarization using pyannote.ai.

**Upload a local audio file:**

```bash
uv run diarization/upload.py path/to/audio.wav
```

**Diarize an audio file (public URL or uploaded media):**

```bash
uv run diarization/diarize.py 'media://your-object-key'
# or with a public URL
uv run diarization/diarize.py 'https://example.com/audio.wav'
```

## Contributing

To avoid merge conflicts, **each contributor should push work in their own folder**.

