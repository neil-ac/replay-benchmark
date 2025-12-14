"""
pip install "transformers[torch]==v4.49.0"
"""

import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}, 
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "/home/eric_bezzam/s2s-evals/model_usage/audio_understanding/1st_topic_discussion.wav"},
        # {"type": "text", "text": "Is there anything strange in this discussion?"},
        {"type": "text", "text": "Are there interruptions in the discussion?"},   # Yes, there are moments when one speaker talks over another.
    ]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(
                    librosa.load(
                        ele['audio_url'], 
                        sr=processor.feature_extractor.sampling_rate)[0]
                )

inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
inputs.input_ids = inputs.input_ids.to("cuda")

generate_ids = model.generate(**inputs, max_length=1000)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


print("Generated response:")
print("=" * 80)
print(response)
print("=" * 80)


"""
Yes, the speakers seem to be discussing something unusual or unexpected, as indicated by their reactions such as 'hmm okay' and laughter.
"""