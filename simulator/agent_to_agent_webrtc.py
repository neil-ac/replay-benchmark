"""
Connect two Pipecat agents locally using serverless WebRTC (SmallWebRTCTransport).
This script creates two agents and connects them directly without any WebRTC server.
"""

import asyncio
import io
import os
import sqlite3
import time
import wave
from datetime import datetime
from aiortc import RTCSessionDescription
from dotenv import load_dotenv
from loguru import logger
import aiofiles
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.gradium.stt import GradiumSTTService
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.frames.frames import (
    LLMTextFrame, TTSTextFrame, TranscriptionFrame, Frame,
    InputAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame, InterimTranscriptionFrame,
    BotStartedSpeakingFrame, BotStoppedSpeakingFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.utils.tracing.setup import setup_tracing
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

load_dotenv(override=True)

IS_TRACING_ENABLED = bool(os.getenv("ENABLE_TRACING"))

if IS_TRACING_ENABLED:
    otlp_exporter = OTLPSpanExporter()
    setup_tracing(
        service_name="simulator-agent-e2e",
        exporter=otlp_exporter,
    )
    logger.info("OpenTelemetry tracing initialized")


# SQLite database path for turn metrics
DB_PATH = "recordings/conversation_turns.db"


def init_database():
    """Initialize SQLite database for turn metrics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_turn (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            turn_start_time REAL,
            turn_end_time REAL,
            agent_speech_text TEXT,
            simulator_response_text TEXT,
            voice_to_voice_response_time REAL,
            interrupted BOOLEAN
        )
    """)
    conn.commit()
    conn.close()


class ConversationFlowLogger(FrameProcessor):
    """Frame processor to log conversation flow for debugging agent-to-agent interactions."""
    
    def __init__(self, agent_name: str):
        super().__init__()
        self._agent_name = agent_name
        self._input_audio_count = 0
        self._transcription_count = 0
    
    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        
        # Log incoming audio from the other agent
        if isinstance(frame, InputAudioRawFrame):
            self._input_audio_count += 1
            if self._input_audio_count % 50 == 0:  # Log every 50 frames (~1 second at 20ms per frame)
                logger.info(f"{self._agent_name}: Received {self._input_audio_count} audio input frames from other agent")
        
        # Log VAD decisions
        elif isinstance(frame, UserStartedSpeakingFrame):
            logger.info(f"{self._agent_name}: VAD detected user started speaking")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(f"{self._agent_name}: VAD detected user stopped speaking")
        
        # Log transcriptions (both interim and final)
        elif isinstance(frame, InterimTranscriptionFrame):
            logger.debug(f"{self._agent_name}: Interim transcription: {frame.text}")
        elif isinstance(frame, TranscriptionFrame):
            self._transcription_count += 1
            logger.info(f"{self._agent_name}: FINAL TRANSCRIPTION #{self._transcription_count}: '{frame.text}'")
        
        # Log LLM responses
        elif isinstance(frame, LLMTextFrame):
            logger.info(f"{self._agent_name}: LLM response: '{frame.text}'")
        
        # Log TTS input
        elif isinstance(frame, TTSTextFrame):
            logger.info(f"{self._agent_name}: TTS will speak: '{frame.text}'")
        
        await self.push_frame(frame, direction)


class TurnTracker(FrameProcessor):
    """Tracks turn metrics including voice-to-voice response time and saves to SQLite."""

    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id
        self.db_connection = sqlite3.connect(DB_PATH)
        self._init_turn_values()

    def _init_turn_values(self):
        self._turn_number = 0
        self._turn_start_time = 0
        self._turn_end_time = 0
        self._simulator_stopped_speaking_ts = 0
        self._voice_to_voice_time = 0
        self._agent_speech_text = ""
        self._simulator_response_text = ""
        self._interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Track voice-to-voice timing for the Agent (not Simulator)
        # From SimulatorAgent's perspective: User = Agent, Bot = Simulator
        # Measure: Simulator stops speaking → Agent starts speaking again
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._simulator_stopped_speaking_ts = time.time()
        elif isinstance(frame, UserStartedSpeakingFrame):
            # Agent started speaking again after Simulator finished responding
            if self._simulator_stopped_speaking_ts > 0:
                self._voice_to_voice_time = time.time() - self._simulator_stopped_speaking_ts
                logger.info(f"Agent voice-to-voice response time: {self._voice_to_voice_time:.3f}s")

        await self.push_frame(frame, direction)

    async def set_agent_speech_text(self, text: str):
        self._agent_speech_text += " " + text

    async def set_simulator_response_text(self, text: str):
        self._simulator_response_text = text
        # If turn has already ended, save now
        if self._turn_start_time:
            await self._save_turn()

    async def end_turn(self, turn_number: int, turn_start_time: float,
                       turn_end_time: float, interrupted: bool):
        self._turn_number = turn_number
        self._turn_start_time = turn_start_time
        self._turn_end_time = turn_end_time
        self._interrupted = interrupted
        # If we have LLM response, save now
        if self._simulator_response_text:
            await self._save_turn()

    async def _save_turn(self):
        """Save turn metrics to SQLite."""
        agent_text_preview = self._agent_speech_text[:50] if self._agent_speech_text else ""
        logger.info(f"Saving turn {self._turn_number}: agent='{agent_text_preview}...', "
                    f"v2v={self._voice_to_voice_time:.3f}s, interrupted={self._interrupted}")
        cursor = self.db_connection.cursor()
        cursor.execute(
            """INSERT INTO conversation_turn
               (session_id, turn_number, turn_start_time, turn_end_time,
                agent_speech_text, simulator_response_text, voice_to_voice_response_time, interrupted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.session_id, self._turn_number, self._turn_start_time,
             self._turn_end_time, self._agent_speech_text, self._simulator_response_text,
             self._voice_to_voice_time, self._interrupted)
        )
        self.db_connection.commit()
        self._init_turn_values()


class ConversationEndDetector(FrameProcessor):
    """Detects when SimulatorAgent says goodbye to end the conversation."""

    def __init__(self, on_end_callback):
        super().__init__()
        self._on_end = on_end_callback
        self._end_phrases = ["goodbye", "bye bye", "bye", "thanks bye", "thank you bye", "have a good"]
        self._triggered = False

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)

        # Monitor LLM output for goodbye phrases
        if isinstance(frame, LLMTextFrame) and not self._triggered:
            text_lower = frame.text.lower().strip()
            for phrase in self._end_phrases:
                if phrase in text_lower:
                    self._triggered = True
                    logger.info(f"ConversationEndDetector: Detected '{phrase}' - will end conversation after TTS completes")
                    # Delay slightly to allow TTS to finish the goodbye
                    asyncio.get_event_loop().call_later(3.0, lambda: asyncio.create_task(self._on_end()))
                    break

        await self.push_frame(frame, direction)


# Check required env vars
if not os.getenv("GRADIUM_API_KEY"):
    raise ValueError("GRADIUM_API_KEY not found in environment. Set it in .env file.")
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment. Set it in .env file.")
if not os.getenv("DEEPGRAM_API_KEY"):
    raise ValueError("DEEPGRAM_API_KEY not found in environment. Set it in .env file.")
if not os.getenv("CARTESIA_API_KEY"):
    raise ValueError("CARTESIA_API_KEY not found in environment. Set it in .env file.")

logger.remove()
# Enable DEBUG level logging to see all frame processing
# Use stderr for better visibility and include all DEBUG+ messages
import sys
logger.add(
    sys.stderr, 
    level="DEBUG", 
    format="{time:HH:mm:ss} | {level: <8} | {name} | {message}",
    colorize=True
)

# Create output directory for audio files
AUDIO_OUTPUT_DIR = "recordings"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)


def create_basic_agent_setup(webrtc_connection: SmallWebRTCConnection, session_id: str, audio_mixer=None):
    """Set up Basic Agent (Deepgram STT + Cartesia TTS) with WebRTC connection."""
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": """You are a pizza ordering agent for Mario's Pizzeria. Your role is to take orders and confirm them with the total.

Menu: Small $10, Medium $14, Large $18. Toppings $1.50 each. Thin crust +$1, Stuffed +$3.

RULES:
- Keep responses short (1-2 sentences max)
- If customer gives complete order (size, toppings, crust), immediately confirm and give total
- Do NOT ask unnecessary questions if info was already provided


Speak in English only. No emojis.""",
        },
        {
            "role": "user",
            "content": "Customer is calling. Greet them and ask them how you can help them today.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)
    agent_name = "BasicAgent"
    
    # Add RTVIProcessor like in the original basic agent
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    
    return _create_agent_pipeline(agent_name, webrtc_connection, session_id, stt, llm, tts, context_aggregator, rtvi, audio_mixer)


def create_simulator_agent_setup(webrtc_connection: SmallWebRTCConnection, session_id: str, audio_mixer=None, on_end_callback=None):
    """Set up Simulator Agent (Gradium STT + Gradium TTS) with WebRTC connection."""
    # Configure Gradium STT for English to avoid transcribing as Spanish
    stt = GradiumSTTService(
        api_key=os.getenv("GRADIUM_API_KEY"),
        json_config='{"language": "en"}'
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
    tts = GradiumTTSService(
        api_key=os.getenv("GRADIUM_API_KEY"),
        voice_id="YTpq7expH9539ERJ",  # A pleasant and smooth female voice ready to assist your customers and also eager to have nice converstations
    )

    context = OpenAILLMContext(
        [
            {
                "role": "system",
                "content": """You are a customer ordering pizza. Be concise and efficient.

YOUR ORDER: One large pizza, pepperoni and mushrooms, thin crust.

RULES:
- Give your uncompleted order in your first response: "I'd like a large thin crust pizza"
- Keep responses to 1 sentence
- When agent confirms order and gives total, say "Perfect, thanks! Goodbye" and nothing else
- Do NOT ask questions or make small talk

Speak in English only. No emojis.""",
            },
        ],
    )

    context_aggregator = llm.create_context_aggregator(context)
    agent_name = "SimulatorAgent"

    # Create end detector if callback provided
    end_detector = ConversationEndDetector(on_end_callback) if on_end_callback else None

    return _create_agent_pipeline(agent_name, webrtc_connection, session_id, stt, llm, tts, context_aggregator, None, audio_mixer, enable_tracing=IS_TRACING_ENABLED, end_detector=end_detector)


def _create_agent_pipeline(agent_name: str, webrtc_connection: SmallWebRTCConnection, session_id: str,
                          stt, llm, tts, context_aggregator, rtvi=None, audio_mixer=None, enable_tracing=False, end_detector=None):
    """Common pipeline setup for both agents.

    For SimulatorAgent: Includes full observability (AudioBufferProcessor, TurnTrackingObserver, TranscriptProcessor, TurnTracker)
    For BasicAgent: Simple pipeline without observability (it's the system under test)
    """
    is_simulator = agent_name == "SimulatorAgent"

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.6)),
        ),
    )

    # Conversation flow logger for debugging agent-to-agent interactions
    conversation_logger = ConversationFlowLogger(agent_name)

    # Observability components (SimulatorAgent only)
    audio_buffer = None
    turn_tracker = None
    transcript_processor = None
    turn_observer = None

    if is_simulator:
        audio_buffer = AudioBufferProcessor()
        turn_tracker = TurnTracker(session_id)
        transcript_processor = TranscriptProcessor()
        turn_observer = TurnTrackingObserver()

    # Add observers for debugging
    observers = []
    if rtvi:
        observers.append(RTVIObserver(rtvi))

    # Add DebugLogObserver to log conversational frames (LLM, STT, TTS)
    debug_observer = DebugLogObserver(
        frame_types=(
            LLMTextFrame,
            TTSTextFrame,
            TranscriptionFrame,
        )
    )
    observers.append(debug_observer)

    # Add TurnTrackingObserver for SimulatorAgent
    if turn_observer:
        observers.append(turn_observer)

    # Build pipeline
    if isinstance(context_aggregator, LLMContextAggregatorPair):
        # Basic agent pipeline order (with RTVIProcessor if provided)
        pipeline_steps = [
            transport.input(),
            conversation_logger,
        ]
        if rtvi:
            pipeline_steps.append(rtvi)
        pipeline_steps.extend([
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])
        pipeline = Pipeline(pipeline_steps)
    else:
        # Simulator agent pipeline order with full observability
        pipeline_steps = [
            transport.input(),
            conversation_logger,
            stt,
        ]
        # Add user transcript processor after STT
        if transcript_processor:
            pipeline_steps.append(transcript_processor.user())
        pipeline_steps.extend([
            context_aggregator.user(),
            llm,
        ])
        # Add end detector after LLM if provided
        if end_detector:
            pipeline_steps.append(end_detector)
        pipeline_steps.append(tts)
        pipeline_steps.append(transport.output())
        # Add assistant transcript processor after TTS output
        if transcript_processor:
            pipeline_steps.append(transcript_processor.assistant())
        # Add turn tracker after transcript
        if turn_tracker:
            pipeline_steps.append(turn_tracker)
        # Add audio buffer at the end to capture full conversation
        if audio_buffer:
            pipeline_steps.append(audio_buffer)
        pipeline_steps.append(context_aggregator.assistant())
        pipeline = Pipeline(pipeline_steps)

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=observers,
        enable_tracing=enable_tracing,
        conversation_id=f"{agent_name}-conversation",
    )

    # Guard flag to prevent duplicate on_client_connected handling
    _client_connected_handled = False

    # Add event handlers for TTS to debug Cartesia connection
    if hasattr(tts, 'event_handler'):
        @tts.event_handler("on_connected")
        async def on_tts_connected(tts_service):
            logger.info(f"{agent_name}: TTS connected")

        @tts.event_handler("on_disconnected")
        async def on_tts_disconnected(tts_service):
            logger.info(f"{agent_name}: TTS disconnected")

        @tts.event_handler("on_connection_error")
        async def on_tts_error(tts_service, error):
            logger.error(f"{agent_name}: TTS connection error: {error}")

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        nonlocal _client_connected_handled

        if _client_connected_handled:
            logger.debug(f"{agent_name}: on_client_connected already handled, skipping")
            return
        _client_connected_handled = True

        logger.info(f"{agent_name}: Client connected")

        # Start audio recording for SimulatorAgent
        if audio_buffer:
            await audio_buffer.start_recording()
            logger.info(f"{agent_name}: Audio recording started")

        # Wait for StartFrame to be processed by the pipeline
        await asyncio.sleep(1.5)

        # Only BasicAgent should speak first
        if agent_name == "BasicAgent":
            logger.info(f"{agent_name}: Sending LLMRunFrame to start conversation")
            await task.queue_frames([LLMRunFrame()])
        else:
            logger.info(f"{agent_name}: Ready to receive audio input (will respond after hearing BasicAgent)")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"{agent_name}: Client disconnected")
        if audio_buffer:
            await audio_buffer.stop_recording()
        await task.cancel()

    # Wire up observability event handlers for SimulatorAgent
    if is_simulator and turn_observer and turn_tracker:
        @turn_observer.event_handler("on_turn_ended")
        async def on_turn_ended(observer, turn_number, duration, was_interrupted):
            end_time = time.time()
            start_time = end_time - duration
            await turn_tracker.end_turn(turn_number, start_time, end_time, was_interrupted)

    if is_simulator and transcript_processor and turn_tracker:
        @transcript_processor.event_handler("on_transcript_update")
        async def on_transcript_update(processor, frame):
            for message in frame.messages:
                if message.role == "user":
                    await turn_tracker.set_agent_speech_text(message.content)
                elif message.role == "assistant":
                    await turn_tracker.set_simulator_response_text(message.content)

    if is_simulator and audio_buffer:
        @audio_buffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            filename = os.path.join(AUDIO_OUTPUT_DIR, f"conversation-{session_id}.wav")
            try:
                async with aiofiles.open(filename, "wb") as f:
                    with io.BytesIO() as buf:
                        with wave.open(buf, "wb") as wf:
                            wf.setsampwidth(2)
                            wf.setnchannels(num_channels)
                            wf.setframerate(sample_rate)
                            wf.writeframes(audio)
                        await f.write(buf.getvalue())
                logger.info(f"Saved conversation recording: {filename} ({len(audio)} bytes)")
            except Exception as e:
                logger.exception(f"Error saving audio: {e}")

    return transport, task


async def run_agent(agent_name: str, transport, task):
    """Run an agent's pipeline."""
    runner = PipelineRunner(handle_sigint=False, force_gc=True)
    await runner.run(task)


async def connect_two_agents():
    """Connect two agents using serverless WebRTC with 60-second timeout and audio recording.

    Audio recording and turn metrics are handled automatically by:
    - AudioBufferProcessor: Records full conversation (input + output) on SimulatorAgent
    - TurnTrackingObserver: Tracks turn metrics (duration, interruptions)
    - TurnTracker: Measures voice-to-voice response time, saves to SQLite
    - TranscriptProcessor: Captures user/assistant transcript text
    """
    # Initialize SQLite database for turn metrics
    init_database()

    # Generate session ID for this conversation
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting conversation session: {session_id}")
    logger.info(f"Audio will be saved to: {AUDIO_OUTPUT_DIR}/conversation-{session_id}.wav")
    logger.info(f"Turn metrics will be saved to: {DB_PATH}")

    # Event to signal graceful conversation end (when SimulatorAgent says goodbye)
    conversation_ended = asyncio.Event()

    async def end_conversation():
        """Called when SimulatorAgent signals the end of the conversation."""
        logger.info("Graceful conversation end triggered by SimulatorAgent")
        conversation_ended.set()

    # Create two connections (no ICE servers needed for localhost)
    ice_servers = []
    conn1 = SmallWebRTCConnection(ice_servers=ice_servers)
    conn2 = SmallWebRTCConnection(ice_servers=ice_servers)

    # Agent 1 creates an offer
    logger.info("Agent 1: Creating offer...")
    conn1.pc.addTransceiver("audio", direction="sendrecv")
    conn1.pc.createDataChannel("signalling")
    offer = await conn1.pc.createOffer()
    await conn1.pc.setLocalDescription(offer)

    logger.info("Agent 1: Offer created, sending to Agent 2...")

    # Agent 2 receives the offer and creates an answer
    logger.info("Agent 2: Receiving offer, creating answer...")
    await conn2.initialize(sdp=offer.sdp, type=offer.type)

    answer_dict = conn2.get_answer()
    if not answer_dict:
        raise Exception("Failed to get answer from agent 2")

    logger.info("Agent 2: Answer created, sending to Agent 1...")

    # Agent 1 receives the answer
    answer = RTCSessionDescription(sdp=answer_dict["sdp"], type=answer_dict["type"])
    await conn1.pc.setRemoteDescription(answer)

    logger.info("Agent 1: Answer received, setting up agents...")

    # Set up both agents
    transport1, task1 = create_basic_agent_setup(conn1, session_id)
    transport2, task2 = create_simulator_agent_setup(conn2, session_id, on_end_callback=end_conversation)

    # Start both agents' pipelines
    logger.info("Starting agent pipelines...")
    agent1_task = asyncio.create_task(run_agent("Agent1", transport1, task1))
    agent2_task = asyncio.create_task(run_agent("Agent2", transport2, task2))

    # Give pipelines a moment to initialize
    await asyncio.sleep(1.0)

    # Connect both agents
    logger.info("Connecting agents...")
    await conn1.connect()
    await conn2.connect()

    # Give agents time to fully initialize after connection
    await asyncio.sleep(1.0)

    CONVERSATION_TIMEOUT = 60.0  # 1 minute max
    logger.info(f"Conversation started. Will timeout after {CONVERSATION_TIMEOUT:.0f} seconds or end when customer says goodbye...")
    start_time = time.time()

    # Wait for either: conversation end (SimulatorAgent says goodbye) OR timeout
    async def wait_for_end_or_timeout():
        """Wait for conversation to end naturally or timeout."""
        end_reason = "unknown"

        end_wait_task = asyncio.create_task(conversation_ended.wait())
        timeout_task = asyncio.create_task(asyncio.sleep(CONVERSATION_TIMEOUT))

        try:
            done, pending = await asyncio.wait(
                [end_wait_task, timeout_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            if end_wait_task in done:
                end_reason = "graceful"
                elapsed = time.time() - start_time
                logger.info(f"Conversation ended gracefully (elapsed: {elapsed:.1f}s)")
            else:
                end_reason = "timeout"
                elapsed = time.time() - start_time
                logger.info(f"{CONVERSATION_TIMEOUT:.0f}-second timeout reached (elapsed: {elapsed:.1f}s). Stopping agents...")

            for t in pending:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        except Exception as e:
            logger.warning(f"Error waiting for conversation end: {e}")
            end_reason = "error"

        return end_reason

    end_reason = await wait_for_end_or_timeout()

    # Cancel both agent tasks
    agent1_task.cancel()
    agent2_task.cancel()

    try:
        await asyncio.gather(agent1_task, agent2_task, return_exceptions=True)
    except Exception:
        pass

    # Disconnect connections
    try:
        await conn1.disconnect()
        await conn2.disconnect()
    except Exception:
        pass

    elapsed = time.time() - start_time
    logger.info(f"Conversation ended ({end_reason}). Total time: {elapsed:.1f}s")
    logger.info(f"Output files:")
    logger.info(f"  - {AUDIO_OUTPUT_DIR}/conversation-{session_id}.wav (mixed conversation)")
    logger.info(f"  - {DB_PATH} (turn metrics in SQLite)")


if __name__ == "__main__":
    try:
        asyncio.run(connect_two_agents())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Error: {e}")

