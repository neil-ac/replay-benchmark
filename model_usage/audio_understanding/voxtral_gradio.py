"""
Gradio interface for Voxtral audio understanding model.

pip install "transformers[torch]==v4.56.1" gradio
"""

import gradio as gr
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from transformers.audio_utils import load_audio
import soundfile as sf
import os
import numpy as np


def understand_audio(audio_input, prompt, start_time, end_time):
    """
    Analyze audio using Voxtral model.
    
    Args:
        audio_input: Can be a file path (str) or tuple (sample_rate, audio_array) from Gradio
        prompt: Text prompt for analysis
        start_time: Start time in seconds
        end_time: End time in seconds (None means end of audio)
    
    Returns:
        Analysis result or error message
    """
    try:
        # Handle Gradio audio input (tuple of sample_rate and audio array)
        if isinstance(audio_input, tuple):
            sample_rate_input, audio_array = audio_input
            sample_rate = 16000  # Target sample rate
            
            # Convert to float32 if needed
            audio_array = np.asarray(audio_array, dtype=np.float32)
            
            # Resample if needed
            if sample_rate_input != sample_rate:
                from librosa import resample
                audio = resample(audio_array, orig_sr=sample_rate_input, target_sr=sample_rate)
            else:
                audio = audio_array
        else:
            # File path input
            sample_rate = 16000
            audio = load_audio(audio_input, sampling_rate=sample_rate)
        
        # Ensure audio is float32
        audio = np.asarray(audio, dtype=np.float32)
        
        print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}, Sample rate: {sample_rate}")
        total_duration = audio.shape[0] / sample_rate
        print(f"Total audio duration (seconds): {total_duration}")

        # Extract audio segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate) if end_time else audio.shape[0]
        
        # Validate bounds
        if start_sample >= audio.shape[0]:
            return f"Error: Start time ({start_time}s) is beyond audio duration ({total_duration}s)"
        if start_sample >= end_sample:
            return f"Error: Start time must be less than end time"
        
        audio = audio[start_sample:end_sample]
        extracted_duration = audio.shape[0] / sample_rate
        print(f"Extracted audio duration (seconds): {extracted_duration}")
        
        # Save to temp file
        temp_audio_path = "temp_audio_gradio.wav"
        sf.write(temp_audio_path, audio, samplerate=sample_rate)

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        repo_id = "mistralai/Voxtral-Mini-3B-2507"

        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, 
            torch_dtype=torch.bfloat16, 
            device_map=device
        )

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "audio", "path": temp_audio_path},
                ],
            }
        ]

        # Generate response
        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(device, dtype=torch.bfloat16)

        outputs = model.generate(**inputs, max_new_tokens=500)
        decoded_outputs = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        # Cleanup
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        return decoded_outputs[0]

    except Exception as e:
        return f"Error: {str(e)}"


def create_demo():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(title="Voxtral Audio Understanding") as demo:
        gr.Markdown("""
        # 🎙️ Voxtral Audio Understanding
        
        Analyze audio files using the Voxtral-Mini model. Ask questions about content, 
        speakers, emotions, and more!
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Audio\nSupports WAV, MP3, and other common formats")
                audio_input = gr.Audio(
                    type="numpy",
                    label="Upload or Record Audio"
                )
                
            with gr.Column():
                gr.Markdown("### Time Range")
                start_time = gr.Slider(
                    minimum=0,
                    maximum=3600,
                    value=0,
                    step=0.5,
                    label="Start time (seconds)"
                )
                end_time = gr.Slider(
                    minimum=0,
                    maximum=3600,
                    value=None,
                    step=0.5,
                    label="End time (seconds)"
                )
        
        with gr.Row():
            prompt = gr.Textbox(
                label="Analysis Prompt",
                value="Is there anything strange in this discussion? Are there multiple speakers? Are they overlapping? What are the emotions? Are there interruptions? Are there awkward pauses?",
                lines=3
            )
        
        with gr.Row():
            submit_btn = gr.Button("Analyze Audio", variant="primary", size="lg")
            clear_btn = gr.Button("Clear", size="lg")
        
        with gr.Row():
            output = gr.Textbox(
                label="Analysis Result",
                lines=8,
                interactive=False
            )
        
        # Examples
        gr.Examples(
            examples=[
                [
                    None,  # audio (user will upload)
                    "Does the speaker sound happy? What emotions do you detect?",
                    0,
                    None,
                ],
                [
                    None,
                    "Are there multiple speakers? Do they overlap? Are there interruptions?",
                    0,
                    30,
                ],
                [
                    None,
                    "What are the main topics discussed? Summarize the conversation.",
                    0,
                    None,
                ],
            ],
            inputs=[audio_input, prompt, start_time, end_time],
            label="Example Prompts"
        )
        
        # Event handlers
        submit_btn.click(
            fn=understand_audio,
            inputs=[audio_input, prompt, start_time, end_time],
            outputs=output
        )
        
        clear_btn.click(
            fn=lambda: (None, "", 0, None, ""),
            outputs=[audio_input, prompt, start_time, end_time, output]
        )
    
    return demo


if __name__ == "__main__":
    # Set up a custom temp directory with proper permissions
    import tempfile
    custom_temp_dir = tempfile.mkdtemp(prefix="voxtral_")
    os.environ["GRADIO_TEMP_DIR"] = custom_temp_dir
    print(f"Using temporary directory: {custom_temp_dir}")
    
    demo = create_demo()
    demo.launch(
        share=True,
        # share=False,
        # server_name="127.0.0.1",  # Only localhost
        server_port=7860,
    )
