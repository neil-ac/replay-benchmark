# Setup

```bash
conda create -n voiceai python=3.10
conda activate voiceai

pip install "pipecat-ai[all]" websockets
pip install "transformers[torch]==v4.56.1"
```

Getting Gradium key: https://eu.api.gradium.ai/studio/
```
export GRADIUM_API_KEY="your_api_key_here"
```
