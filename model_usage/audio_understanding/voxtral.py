"""
pip install "transformers[torch]==v4.56.1"

way more verbose than Flamingo but definitely better!
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
            # {"type": "text", "text": "What's going on in this audio file?"}, 
            # {"type": "text", "text": "What are they talking about?"}, 

            {"type": "text", "text": "Is there anything strange in this discussion?"}, 
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


"""

Generated response:
================================================================================
There are a few things that might be considered strange or unusual in this discussion:

1. **Repetition and Clarification**: The conversation includes several instances where one of the speakers repeats or clarifies what they said earlier. For example:
   - "Sorry, I didn't quite catch that. Did you say it?"
   - "I was just about to say it's easy to fall into that trap of assuming everyone feels the same way about a city. What makes you say that?"
   - "You're right to call me out on that. I did trail off. My bet."

2. **Pauses and Hesitations**: There are several pauses and hesitations in the conversation, which can sometimes make it feel a bit disjointed. For instance:
   - "Hmm? Okay. Sorry, I didn't."
   - "Could you say it again?"
   - "I was just about to say it's easy to fall into that trap of assuming everyone feels the same way about a city."

3. **The Reference to a Website**: At the end of the conversation, one of the speakers mentions creating a free account on the Sesame website to get longer calls. This is a bit unusual, as it's not directly related to the topic of Munich or the discussion about the city's evolution and cultural identity.

4. **The Ending**: The conversation ends with a bit of a strange exchange, where one speaker says "You too" twice, and the other responds with "You as well." This exchange feels a bit abrupt and not entirely natural, especially given the context of the conversation.

5. **The Reference to Beer Gardens and Traditional Costumes**: While it's not strange in itself, the speaker's mention of beer gardens and traditional costumes as the only things Munich is known for is a bit reductive. The conversation later acknowledges that Munich is more complex than that, which could be seen as a bit of a contradiction.

Overall, these elements might make the conversation feel a bit disjointed or unusual, but they are not necessarily strange in a negative way. They could be seen as indicative of a natural, conversational flow, with pauses, repetitions, and clarifications being common in real-life conversations.
================================================================================

"""
