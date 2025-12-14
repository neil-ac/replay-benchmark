"""
pip install "transformers[torch]==v5.0.0rc0"
"""




from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            # {"type": "text", "text": "Transcribe the input speech."},
            # {"type": "audio", "path": "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/WhDJDIviAOg_120_10.mp3"},
            
            # {"type": "text", "text": "What's going on in this audio file?"}, 
            # RESPONSE: A conversation between two people is taking place.
            # {"type": "text", "text": "What are they talking about?"}, 
            # RESPONSE: They are discussing the cultural significance and evolution of Munich, including its historical context and identity.
            # {"type": "text", "text": "Is there anything strange in this dicussion?"},
            # RESPONSE: No, there is nothing strange in this dicussion.
            {"type": "text", "text": "Is there anything strange in this dicussion? Like people getting interrupted? If so, who is interrupting the most?"},
            # RESPONSE: Yes, there are interruptions. The interviewer interrupts the most, asking questions and prompting the interviewee to elaborate.
            {"type": "audio", "path": "/home/eric_bezzam/s2s-evals/model_usage/audio_understanding/1st_topic_discussion.wav"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1000)

decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(decoded_outputs)
