"""
pip install "transformers[torch]==v4.56.1"

way more versbose than Flamingo but definitely better!
"""

from transformers import VoxtralForConditionalGeneration, AutoProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "mistralai/Voxtral-Mini-3B-2507"

processor = AutoProcessor.from_pretrained(repo_id)
model = VoxtralForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16, device_map=device)

conversation = [
    {
        "role": "user",
        "content": [
            # {"type": "text", "text": "What's going on in this audio file?"}, # A conversation between two people is taking place.
            # {"type": "text", "text": "What are they talking about?"}, # They are discussing the cultural significance and evolution of Munich, including its historical context and identity.
            {"type": "text", "text": "Is there anything strange in this discussion?"}, #No, there is nothing strange in this discussion.
            {"type": "audio", "path": "/home/eric_bezzam/s2s-evals/model_usage/audio_understanding/1st_topic_discussion.wav"},
        ],
    }
]

inputs = processor.apply_chat_template(conversation)
inputs = inputs.to(device, dtype=torch.bfloat16)

outputs = model.generate(**inputs, max_new_tokens=500)
decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\nGenerated response:")
print("=" * 80)
print(decoded_outputs[0])
print("=" * 80)
