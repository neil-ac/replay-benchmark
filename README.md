# Replay - The first open source framework for e2e benchmark of voice applications

An open source simulator-driven evaluation setup for end-to-end testing of voice-to-voice applications.

![Project Screenshot](./images/s2s-evals.png)

Currently, there are no good setups available to test voice-to-voice apps comprehensively, especially for latency, interruption handling, and other production critical aspects. This project fills that gap by enabling you to test any agent against another agent at scale for production use.

## Architecture

Replay consists of two main components: a **simulator** for running conversations and an **analysis pipeline** for evaluating conversation quality.

### Simulator

The simulator orchestrates real-time voice conversations between agents deployed on Pipecat Cloud. It uses the Pipecat Cloud Session API to instantiate both a simulator agent and your tested agent, connecting them to the same Daily.co room for bidirectional audio communication. The simulator handles room creation, token generation, and session management, allowing agents to interact naturally while recording the full conversation audio.

Key features:

- **Cloud-native deployment**: Agents run on Pipecat Cloud infrastructure, enabling scalable testing
- **Real-time audio**: Uses Daily.co WebRTC for low-latency voice communication
- **Session orchestration**: Programmatically starts and manages agent sessions via Pipecat Cloud API
- **Room coordination**: Ensures both agents connect to the same room by passing room URLs via session parameters

### Analysis Pipeline

The analysis pipeline processes recorded conversation audio to extract quantitative metrics about conversation quality. It uses speaker diarization (pyannote.ai) to segment audio by speaker, then analyzes three critical dimensions:

1. **Latency Analysis**: Measures response times between speaker turns, identifying slow responses (>2s threshold) and intra-speaker pauses. Uses Voxtral (audio LLM) to classify pauses as natural conversational pauses vs unnatural processing delays.

2. **Overlap Analysis**: Detects interruptions and simultaneous speech by identifying temporal overlaps between speakers. Analyzes overlap duration and uses Voxtral to distinguish natural backchanneling from problematic interruptions.

3. **Repetition Analysis**: Identifies consecutive n-gram repetitions (1-10 grams) within speaker segments, flagging potential stuttering, processing errors, or system failures.

The pipeline outputs consolidated metrics including event counts, average latencies, natural vs unnatural classifications, and detailed event timelines. All analysis uses audio-level understanding via Voxtral rather than transcription, enabling evaluation of prosody, timing, and naturalness that text-based metrics miss.

## Simulator

End-to-end voice evaluation simulator for testing agents in conversation. Deploy both a simulator agent and your tested agent to Pipecat Cloud, then use `run_conversation.py` to put them in the same Daily room for conversation testing.

**Quick start:**

1. Deploy simulator and tested agents to Pipecat Cloud (see `simulator/README.md` for details)
2. Configure environment variables in `simulator/.env`
3. Run the conversation:

```bash
cd simulator
uv run run_conversation.py
```

See [`simulator/README.md`](simulator/README.md) for detailed setup and deployment instructions.

Script:
```bash
uv run final_submission/main.py audio/convo.wav --output results.json
```

Gradio demo:
```bash
uv run s2s_eval_gradio.py
```
Note: will work (way) faster if you have a cuda or mps devices.

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
