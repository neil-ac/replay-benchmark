# Simulator

End-to-end voice evaluation simulator for testing agents in conversation.

## Overview

This simulator allows you to test your agent by putting it in a conversation with a simulator agent. Both agents are deployed to Pipecat Cloud and connected to the same Daily room where they can speak to each other.

## Prerequisites

1. **Pipecat Cloud account**: [Sign up](https://pipecat.daily.co/sign-up)
2. **Docker**: Install [Docker](https://www.docker.com/) and create a [Docker Hub](https://hub.docker.com/) account
3. **Pipecat CLI**: Install with `uv tool install pipecat-ai-cli` (or use `pc` alias)
4. **Daily API key**: Get from [Daily Dashboard](https://dashboard.daily.co/)
5. **Pipecat Public API key**: Get from [Pipecat Cloud Dashboard](https://cloud.pipecat.ai/) (must start with `pk_`)

## Setup

1. Copy `env.example` to `.env` and fill in your API keys:

```bash
cp env.example .env
```

Required environment variables:

- `DEEPGRAM_API_KEY`: For speech-to-text
- `OPENAI_API_KEY`: For LLM
- `CARTESIA_API_KEY`: For text-to-speech
- `DAILY_API_KEY`: For Daily room management
- `PIPECAT_PUBLIC_API_KEY`: Your Pipecat Cloud public API key (starts with `pk_`)
- `DAILY_ROOM_URL`: URL of an existing Daily room (e.g., `https://gradium-hack.daily.co/<room-id>`)

2. Authenticate with Pipecat Cloud:

```bash
pipecat cloud auth login
```

## 1. Initialize and Deploy Agents

### Deploy Simulator Agent

The simulator agent is located in `simulator-agent/basic-agent/`.

1. Navigate to the simulator agent directory:

```bash
cd simulator-agent/basic-agent
```

2. Update `pcc-deploy.toml` with your Docker Hub username:

```toml
agent_name = "simulator-basic-agent"
image = "YOUR_DOCKERHUB_USERNAME/simulator-basic-agent:0.1"
secret_set = "simulator-basic-agent-secrets"
```

3. Create a `.env` file with your API keys (copy from `env.example`)

4. Upload secrets to Pipecat Cloud:

```bash
pipecat cloud secrets set simulator-basic-agent-secrets --file .env
```

5. Build and push Docker image:

```bash
pipecat cloud docker build-push
```

6. Deploy to Pipecat Cloud:

```bash
pipecat cloud deploy
```

### Deploy Tested Agent

The agent you want to test is located in `tested-agents/basic-agent/`.

1. Navigate to the tested agent directory:

```bash
cd tested-agents/basic-agent
```

2. Update `pcc-deploy.toml` with your Docker Hub username:

```toml
agent_name = "basic-agent"
image = "YOUR_DOCKERHUB_USERNAME/basic-agent:0.1"
secret_set = "basic-agent-secrets"
```

3. Create a `.env` file with your API keys

4. Upload secrets to Pipecat Cloud:

```bash
pipecat cloud secrets set basic-agent-secrets --file .env
```

5. Build and push Docker image:

```bash
pipecat cloud docker build-push
```

6. Deploy to Pipecat Cloud:

```bash
pipecat cloud deploy
```

### Important: Agent Code Requirements

**Both agents MUST read the room URL from `args.body`**, not from `args.room_url`. This is critical because `run_conversation.py` passes the room URL via `SessionParams.data`, which becomes available in `args.body`.

Your agent's `cloud_bot` function should look like this:

```python
async def cloud_bot(args: SessionArguments):
    if isinstance(args, DailySessionArguments):
        # Parse room_url from args.body (highest priority)
        room_url = None
        token = args.token

        if args.body is not None:
            import json
            body_dict = args.body
            if isinstance(args.body, str):
                body_dict = json.loads(args.body)

            if isinstance(body_dict, dict):
                body_room_url = body_dict.get("room_url")
                body_token = body_dict.get("token")
                if body_room_url:
                    room_url = body_room_url
                    token = body_token or token

        # Fall back to env var or args.room_url if needed
        if not room_url:
            room_url = os.getenv("DAILY_ROOM_URL") or args.room_url

        transport = DailyTransport(room_url, token, "Your Agent Name", ...)
        await run_bot(transport, None, agent_name="YourAgent")
```

## 2. Run Conversation

Once both agents are deployed, use `run_conversation.py` to put them in the same room:

1. Set environment variables in your `.env`:

```bash
AGENT_A_NAME=basic-agent          # Your tested agent name
AGENT_B_NAME=simulator-basic-agent # Simulator agent name
DAILY_ROOM_URL=https://gradium-hack.daily.co/<room-id>
```

2. Run the conversation script:

```bash
python run_conversation.py
```

The script will:

1. Create a Daily room token for the specified room
2. Start Agent A (your tested agent) in the room
3. Start Agent B (simulator agent) in the same room
4. Keep both agents connected so they can converse

Press `Ctrl+C` to stop the conversation.

### How It Works

`run_conversation.py` uses the Pipecat Cloud Session API to start both agents:

```python
from pipecatcloud.session import Session, SessionParams

# Create session parameters with room URL in data
params = SessionParams(
    use_daily=True,
    data={
        "room_url": room_url,
        "token": token,
        "agent_initiator": "true"  # or "false"
    }
)

# Start the agent session
session = Session(
    agent_name="your-agent-name",
    api_key=PIPECAT_PUBLIC_API_KEY,
    params=params
)

await session.start()
```

The `data` parameter is passed to your agent's `cloud_bot` function as `args.body`, which is why agents must read from `args.body` instead of `args.room_url`.

## Troubleshooting

- **Agents in separate rooms**: Verify your agent code reads `room_url` from `args.body`, not `args.room_url`
- **Agent not starting**: Check agent logs in Pipecat Cloud dashboard
- **API key errors**: Ensure you're using a PUBLIC API key (starts with `pk_`), not a secret key
- **Room URL errors**: Make sure `DAILY_ROOM_URL` is set correctly in your `.env` file
