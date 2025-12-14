import os
import asyncio
import requests
import time
import sys
import json
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "").strip()
# Try PIPECAT_PUBLIC_API_KEY first, then fall back to PIPECAT_API_KEY
PIPECAT_PUBLIC_API_KEY = os.getenv("PIPECAT_PUBLIC_API_KEY", "").strip() or os.getenv("PIPECAT_API_KEY", "").strip()
AGENT_A_NAME = os.getenv("AGENT_A_NAME", "basic-agent").strip()
AGENT_B_NAME = os.getenv("AGENT_B_NAME", "simulator-basic-agent").strip()

# Required: Use existing room URL from environment variable
DAILY_ROOM_URL = os.getenv("DAILY_ROOM_URL", "").strip()

if not DAILY_API_KEY:
    print("Error: DAILY_API_KEY environment variable is missing.")
    sys.exit(1)

if not DAILY_ROOM_URL:
    print("Error: DAILY_ROOM_URL environment variable is missing.")
    print("Set DAILY_ROOM_URL to use an existing Daily room.")
    sys.exit(1)

if not PIPECAT_PUBLIC_API_KEY:
    print("Error: PIPECAT_PUBLIC_API_KEY or PIPECAT_API_KEY environment variable is missing.")
    print("Get your public API key from: https://cloud.pipecat.ai/")
    print("Note: You need a PUBLIC API key (starts with 'pk_'), not a secret key (starts with 'sk_')")
    sys.exit(1)

# Validate that it's a public API key
if PIPECAT_PUBLIC_API_KEY.startswith("sk_"):
    print("Error: You're using a SECRET API key (starts with 'sk_'), but the Session API requires a PUBLIC API key (starts with 'pk_').")
    print("Get your public API key from: https://cloud.pipecat.ai/")
    print("Set it as PIPECAT_PUBLIC_API_KEY in your .env file")
    sys.exit(1)

if not PIPECAT_PUBLIC_API_KEY.startswith("pk_"):
    print("Warning: API key doesn't start with 'pk_'. Make sure you're using a public API key.")


def create_daily_room() -> tuple[str, str]:
    """Use existing Daily room from DAILY_ROOM_URL and return (room_url, token)."""
    print(f"Using existing Daily room: {DAILY_ROOM_URL}")
    # Generate a token for the existing room
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
        "Content-Type": "application/json"
    }
    # Extract room name from URL
    room_name = DAILY_ROOM_URL.split("/")[-1]
    token_url = f"https://api.daily.co/v1/meeting-tokens"
    token_data = {
        "properties": {
            "room_name": room_name,
            "is_owner": True,
            "exp": int(time.time()) + 3600  # 1 hour expiry
        }
    }
    response = requests.post(token_url, json=token_data, headers=headers)
    response.raise_for_status()
    token = response.json()["token"]
    return DAILY_ROOM_URL, token


async def start_agent_session(agent_name: str, room_url: str, token: str, is_initiator: bool = False):
    """Start a Pipecat Cloud agent session.
    
    Note: Agents must read room_url from args.body (SessionParams.data) and use it
    instead of args.room_url which is the room Pipecat Cloud creates.
    
    IMPORTANT: Pipecat Cloud will ALWAYS create a room when use_daily=True.
    The agents MUST ignore args.room_url and use args.body.room_url instead.
    Make sure agents are deployed with code that reads from args.body!
    """
    try:
        from pipecatcloud.session import Session, SessionParams
        from pipecatcloud.exception import AgentStartError
        
        # Prepare data to pass to agent - this will be available in args.body
        agent_data = {
            "room_url": room_url,
            "token": token,
            "agent_initiator": "true" if is_initiator else "false",
        }
        
        print(f"\n   📤 Starting {agent_name} with:")
        print(f"      Target room: {room_url}")
        print(f"      Data being passed: {json.dumps(agent_data, indent=6)}")
        
        # Pass room URL via data parameter - agents need to read this from args.body
        # We keep use_daily=True to get DailySessionArguments, but agents should
        # prioritize room_url from args.body over args.room_url
        params = SessionParams(
            use_daily=True,  # Required to get DailySessionArguments
            data=agent_data
        )
        
        session = Session(
            agent_name=agent_name,
            api_key=PIPECAT_PUBLIC_API_KEY,
            params=params
        )
        
        result = await session.start()
        print(f"   ✅ Session started")
        
        if "dailyRoom" in result:
            pipecat_room = result['dailyRoom']
            print(f"   ⚠️  WARNING: Pipecat Cloud created room: {pipecat_room}")
            print(f"   ✅ Agent MUST IGNORE this and use: {room_url}")
            print(f"   📋 Verify agent logs show: 'Using room URL from body: {room_url}'")
            print(f"   ❌ If agent logs show '{pipecat_room}', agent code needs update!")
        
        return result
    except ImportError:
        print("Error: pipecatcloud package not installed. Install with: pip install pipecatcloud")
        raise
    except Exception as e:
        print(f"Error starting {agent_name}: {e}")
        raise


async def main():
    print("=" * 70)
    print("⚠️  IMPORTANT: Agents MUST be deployed with updated code!")
    print("   Agents must read room_url from args.body, not args.room_url")
    print("   Check agent logs to verify they're using the correct room")
    print("=" * 70)
    
    print("\n1. Setting up Daily room...")
    room_url, token = create_daily_room()
    
    print(f"\n2. Starting Agent A ({AGENT_A_NAME})...")
    session_a = await start_agent_session(AGENT_A_NAME, room_url, token, is_initiator=True)
    
    print(f"\n3. Starting Agent B ({AGENT_B_NAME})...")
    # Small delay to ensure Agent A is ready
    await asyncio.sleep(2)
    session_b = await start_agent_session(AGENT_B_NAME, room_url, token, is_initiator=False)
    
    print("\n--- Conversation Started ---")
    print(f"Room URL: {room_url}")
    print("Agents are now connected and can speak to each other.")
    print("Press Ctrl+C to stop.")
    
    try:
        # Keep the script running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping agents...")
        # Note: Pipecat Cloud sessions will end when agents disconnect from the room
        print("Agents will disconnect when they leave the room.")


if __name__ == "__main__":
    asyncio.run(main())
