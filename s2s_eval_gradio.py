"""Gradio interface for S2S evaluation pipeline.

This script provides a web UI for the S2S (Speaker-to-Speaker) evaluation pipeline.
Upload an audio file and get consolidated metrics from latency, overlap, and repetition analyses.

Usage:
    python s2s_eval_gradio.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import json
import tempfile
import os
from pathlib import Path
import sys

# Add the final_submission directory to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "final_submission"))

from final_submission.main import run_pipeline


def analyze_audio(audio_file):
    """
    Analyze audio using the S2S evaluation pipeline.
    
    Args:
        audio_file: Audio file from Gradio (tuple of sample_rate and audio_array, or path)
    
    Returns:
        Tuple of (status_message, latency_summary, overlap_summary, repetition_summary)
    """
    try:
        # Handle Gradio audio input
        if isinstance(audio_file, tuple):
            sample_rate, audio_array = audio_file
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio_array, samplerate=sample_rate)
                audio_path = tmp.name
        else:
            audio_path = audio_file
        
        print(f"Analyzing audio: {audio_path}")
        
        # Run the S2S evaluation pipeline
        results = run_pipeline(
            audio_path=audio_path,
            output_path=None,
            latency_threshold=2.0,
            overlap_threshold=0.2,
        )
        
        # Extract summaries from results
        latency_summary = results.get("latency", {}).get("summary", {})
        overlap_summary = results.get("overlap", {}).get("summary", {})
        repetition_summary = results.get("repetition", {}).get("summary", {})
        
        # Format latency summary
        latency_text = format_latency_summary(latency_summary)
        
        # Format overlap summary
        overlap_text = format_overlap_summary(overlap_summary)
        
        # Format repetition summary
        repetition_text = format_repetition_summary(repetition_summary)
        
        status = "✅ Analysis completed successfully!"
        
        # Cleanup temp file if created
        if isinstance(audio_file, tuple) and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        
        return status, latency_text, overlap_text, repetition_text
    
    except Exception as e:
        error_msg = f"❌ Error during analysis: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, "N/A", "N/A", "N/A"


def format_latency_summary(summary):
    """Format latency summary for display."""
    if not summary:
        return "No latency data"
    
    text = "## Latency Analysis\n\n"
    text += f"**Total Events:** {summary.get('total_events', 0)}\n"
    text += f"**Slow Responses:** {summary.get('slow_responses', 0)}\n"
    text += f"**Intra-speaker Pauses:** {summary.get('intra_speaker_pauses', 0)}\n\n"
    
    natural_count = summary.get('natural_count', {})
    text += "**Natural Pauses (by speaker):**\n"
    for speaker, count in natural_count.items():
        text += f"  - {speaker}: {count}\n"
    
    unnatural_count = summary.get('unnatural_count', {})
    text += "\n**Unnatural Pauses (by speaker):**\n"
    for speaker, count in unnatural_count.items():
        text += f"  - {speaker}: {count}\n"
    
    avg_latency = summary.get('average_latency', {})
    text += "\n**Average Latency (seconds, by speaker):**\n"
    for speaker, latency in avg_latency.items():
        text += f"  - {speaker}: {latency:.2f}s\n"
    
    max_latency = summary.get('max_latency', {})
    text += "\n**Max Latency (seconds, by speaker):**\n"
    for speaker, latency in max_latency.items():
        text += f"  - {speaker}: {latency:.2f}s\n"
    
    return text


def format_overlap_summary(summary):
    """Format overlap summary for display."""
    if not summary:
        return "No overlap data"
    
    text = "## Overlap Analysis\n\n"
    text += f"**Total Overlaps:** {summary.get('total_overlaps', 0)}\n\n"
    
    natural_count = summary.get('natural_count', {})
    text += "**Natural Overlaps (by speaker):**\n"
    for speaker, count in natural_count.items():
        text += f"  - {speaker}: {count}\n"
    
    unnatural_count = summary.get('unnatural_count', {})
    text += "\n**Unnatural Overlaps (by speaker):**\n"
    for speaker, count in unnatural_count.items():
        text += f"  - {speaker}: {count}\n"
    
    return text


def format_repetition_summary(summary):
    """Format repetition summary for display."""
    if not summary:
        return "No repetition data"
    
    text = "## Repetition Analysis\n\n"
    text += f"**Total Repetitions:** {summary.get('total_repetitions', 0)}\n\n"
    
    natural_count = summary.get('natural_count', {})
    text += "**Natural Repetitions (by speaker):**\n"
    for speaker, count in natural_count.items():
        text += f"  - {speaker}: {count}\n"
    
    unnatural_count = summary.get('unnatural_count', {})
    text += "\n**Unnatural Repetitions (by speaker):**\n"
    for speaker, count in unnatural_count.items():
        text += f"  - {speaker}: {count}\n"
    
    return text


def create_demo():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="S2S Evaluation Pipeline") as demo:
        gr.Markdown("""
        # 🎙️ Speaker-to-Speaker (S2S) Evaluation Pipeline
        
        Analyze conversation audio to evaluate:
        - **Latency**: Response times and natural pauses
        - **Overlap**: Speaker interruptions and simultaneous speech
        - **Repetition**: Repeated words and phrases
        
        Upload an audio file and get instant analysis results!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Audio")
                audio_input = gr.Audio(
                    type="numpy",
                    label="Audio File",
                )
                analyze_btn = gr.Button("🔍 Analyze Audio", variant="primary", size="lg")
        
        status_output = gr.Textbox(
            label="Status",
            interactive=False,
            lines=1
        )
        
        with gr.Row():
            with gr.Column():
                latency_output = gr.Markdown(
                    label="Latency Summary",
                    value="Upload audio and click Analyze to see results..."
                )
            
            with gr.Column():
                overlap_output = gr.Markdown(
                    label="Overlap Summary",
                    value="Upload audio and click Analyze to see results..."
                )
        
        with gr.Row():
            repetition_output = gr.Markdown(
                label="Repetition Summary",
                value="Upload audio and click Analyze to see results..."
            )
        
        # Event handler
        analyze_btn.click(
            fn=analyze_audio,
            inputs=audio_input,
            outputs=[status_output, latency_output, overlap_output, repetition_output]
        )
    
    return demo


if __name__ == "__main__":
    # Set up a custom temp directory with proper permissions
    import tempfile
    custom_temp_dir = tempfile.mkdtemp(prefix="s2s_eval_")
    os.environ["GRADIO_TEMP_DIR"] = custom_temp_dir
    print(f"Using temporary directory: {custom_temp_dir}")
    
    demo = create_demo()
    demo.launch(
        share=True
    )
