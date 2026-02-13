"""
WildChat Dataset Processor
Processes WildChat conversations into the expected format for process_raw.py

Original dataset: https://huggingface.co/datasets/allenai/WildChat-1M

WILDCHAT INPUT SCHEMA:
----------------------
Each row in the WildChat dataset represents a complete conversation with the following structure:

Top-level fields:
- conversation_hash (string): Hash of the conversation content (used as post_id)
- model (string): OpenAI model used (e.g., "gpt-4-0314", "gpt-3.5-turbo-0301")
- timestamp (datetime): Timestamp of the last turn in the conversation
- turn (int): Total number of turns in the conversation
- language (string): Most frequently detected language
- state (string): State inferred from IP address (e.g., "Texas", "Barcelona")
- country (string): Country inferred from IP address (e.g., "United States", "Spain")
- hashed_ip (string): Hashed IP address (used as user_id)
- header (dict): Request headers with user-agent and accept-language
- toxic (bool): Whether conversation contains toxic content
- redacted (bool): Whether PII was detected and anonymized
- openai_moderation (list): OpenAI moderation results
- detoxify_moderation (list): Detoxify moderation results

conversation (list): List of conversation turns, each turn is a dict with:
  For user turns:
    - role (string): "user"
    - content (string): User's message text
    - hashed_ip (string): Hashed IP address
    - state (string): State from IP
    - country (string): Country from IP
    - header (dict): Request headers
    - language (string): Detected language
    - redacted (bool): Whether PII was detected
    - toxic (bool): Whether content is toxic
    - turn_identifier (int): Unique identifier for the turn
    - timestamp (None): User turns don't have timestamps
  
  For assistant turns:
    - role (string): "assistant"
    - content (string): Assistant's response text
    - timestamp (datetime): When the response was received
    - language (string): Detected language
    - redacted (bool): Whether PII was detected
    - toxic (bool): Whether content is toxic
    - turn_identifier (int): Unique identifier for the turn
    - hashed_ip, state, country, header: None (not present for assistant turns)

RAW DATA OUTPUT SCHEMA:
------------------------
Each training example has the following structure:

- prompt (list): Conversation history before the user's response
  Each item in the list is a dict with:
    - role (string): 
      * For user turns: hashed_ip (e.g., "22fd87ba9b98f3d379b23c7b52961f...")
      * For assistant turns: model name (e.g., "gpt-4-0314")
    - content (string): The message text
    - metadata (string): JSON string containing:
      * For user turns: language, state, country, redacted, toxic, turn_identifier
      * For assistant turns: model, language, redacted, toxic, turn_identifier

- completion (string): The user's response message

- post_id (string): conversation_hash from WildChat

- user_id (string): hashed_ip from user turns (or "unknown_user" if missing)

- timestamp (int64): Unix timestamp (seconds since epoch)
  * For user turns: Uses timestamp from previous assistant turn, or conversation timestamp

- metadata (string): JSON string containing:
  - language: Detected language
  - state: State from IP address
  - country: Country from IP address
  - redacted: Whether PII was detected (bool)
  - toxic: Whether content is toxic (bool)
  - turn_identifier: Unique identifier for the turn (int)
  - conversation_hash: Hash of the conversation (string)

PROCESSING LOGIC:
-----------------
- Each user turn (after the first) becomes a training example
- First user turn is skipped (no assistant response before it)
- Prompt accumulates all previous turns (user + assistant)
- Completion is the user's response
- Roles use hashed_ip for users and model name for assistants
"""

import argparse
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from datasets import Dataset, Features, Value, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from .utils_parser import memory_usage

FEATURES = Features(
    {
        "prompt": [
            {
                "content": Value(dtype="string", id=None),
                "metadata": Value(dtype="string", id=None),
                "role": Value(dtype="string", id=None),
            }
        ],
        "completion": Value(dtype="string", id=None),
        "post_id": Value(dtype="string", id=None),
        "user_id": Value(dtype="string", id=None),
        "timestamp": Value(dtype="int64", id=None),
        "metadata": Value(dtype="string", id=None),
    }
)


def create_user_id(hashed_ip: Optional[str]) -> str:
    """
    Create a user_id from hashed_ip.
    If hashed_ip is missing, return "unknown_user".
    """
    if hashed_ip:
        return hashed_ip
    else:
        return "unknown_user"


def convert_timestamp_to_unix(timestamp) -> Optional[int]:
    """Convert timestamp to Unix timestamp (seconds since epoch)."""
    if timestamp is None:
        return None
    if isinstance(timestamp, (int, float)):
        # Already a timestamp
        if timestamp > 1e10:  # Milliseconds
            return int(timestamp / 1000)
        return int(timestamp)
    elif hasattr(timestamp, "timestamp"):
        # datetime object
        return int(timestamp.timestamp())
    else:
        return None


def process_conversation(conversation_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a WildChat conversation into training examples.
    
    Each user turn (after the first) becomes a training example where:
    - prompt = all previous turns (user + assistant) in the conversation
    - completion = the user's response
    - role in prompt items = hashed_ip for user turns, model name for assistant turns
    """
    entries = []
    conversation = conversation_row["conversation"]
    conversation_hash = conversation_row["conversation_hash"]
    model = conversation_row.get("model", "unknown")
    top_level_timestamp = conversation_row.get("timestamp")
    
    # Get user_id from the first user turn (all user turns in a conversation should have the same IP)
    user_id = None
    for turn in conversation:
        if turn["role"] == "user" and turn.get("hashed_ip"):
            user_id = create_user_id(turn.get("hashed_ip"))
            break
    
    # If no user_id found, try top-level hashed_ip
    if not user_id:
        user_id = create_user_id(conversation_row.get("hashed_ip"))
    
    # Build training examples: for each user turn (after the first), create an example
    prompt_history = []
    first_user_turn = True
    last_assistant_timestamp = None  # Track timestamp from previous assistant turn
    
    for i, turn in enumerate(conversation):
        if not turn.get("language") == "English":
            break

        if turn["content"].strip() == "":
            # Skip empty user messages
            break
                
        if not turn["role"] == "assistant":
            # Get hashed_ip for this user turn (use as role)
            turn_user_id = create_user_id(turn.get("hashed_ip"))

            # Skip the first user turn (no assistant response before it)
            if not first_user_turn:
                # Create a training example for this user turn
                user_metadata = {
                    "language": turn.get("language"),
                    "state": turn.get("state"),
                    "country": turn.get("country"),
                    "redacted": turn.get("redacted", False),
                    "toxic": turn.get("toxic", False),
                    "turn_identifier": turn.get("turn_identifier"),
                    "conversation_hash": conversation_hash,
                }
                # Remove None values
                user_metadata = {k: v for k, v in user_metadata.items() if v is not None}
                
                # Get timestamp - user turns don't have timestamps
                # Use the timestamp from the previous assistant turn, or fallback to conversation timestamp
                timestamp = last_assistant_timestamp or convert_timestamp_to_unix(top_level_timestamp) or 0
                
                entry = {
                    "prompt": prompt_history.copy(),  # Copy to avoid mutation
                    "completion": turn["content"],  # User's message as completion
                    "post_id": conversation_hash,
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "metadata": json.dumps(user_metadata),
                }
                
                entries.append(entry)
            
            # Add user turn to prompt history with hashed_ip as role
            user_metadata = {
                "language": turn.get("language"),
                "state": turn.get("state"),
                "country": turn.get("country"),
                "redacted": turn.get("redacted", False),
                "toxic": turn.get("toxic", False),
                "turn_identifier": turn.get("turn_identifier"),
            }
            # Remove None values
            user_metadata = {k: v for k, v in user_metadata.items() if v is not None}
            
            prompt_history.append({
                "role": turn_user_id,  # Use hashed_ip as role
                "content": turn["content"],
                "metadata": json.dumps(user_metadata),
            })
            
            first_user_turn = False
        
        elif turn["role"] == "assistant":
            # Store timestamp from assistant turn for next user turn
            last_assistant_timestamp = convert_timestamp_to_unix(turn.get("timestamp"))
            
            # Add assistant turn to prompt history with model name as role
            assistant_metadata = {
                "model": model,
                "language": turn.get("language"),
                "redacted": turn.get("redacted", False),
                "toxic": turn.get("toxic", False),
                "turn_identifier": turn.get("turn_identifier"),
            }
            # Remove None values
            assistant_metadata = {k: v for k, v in assistant_metadata.items() if v is not None}
            
            prompt_history.append({
                "role": model,  # Use model name as role
                "content": turn["content"],
                "metadata": json.dumps(assistant_metadata),
            })
    
    return entries


class WildChatDataset:
    """WildChat dataset processor"""

    def __init__(
        self,
        push_to_hub: Optional[str] = None,
        config_path: Optional[str] = None,
        max_conversations: Optional[int] = None,
    ):
        self.config_path = config_path
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()
        
        self.push_to_hub = push_to_hub
        self.max_conversations = max_conversations

    async def create_raw_dataset(self):
        """Load WildChat and convert to training format"""
        print(f"[{datetime.now()}] Loading WildChat dataset...")
        
        # Load dataset
        if self.max_conversations:
            split = f"train[:{self.max_conversations}]"
            print(f"[{datetime.now()}] Limiting to {self.max_conversations} conversations for testing")
        else:
            split = "train"
        
        ds = load_dataset("allenai/WildChat-1M", split=split, verification_mode="no_checks")
        
        print(f"[{datetime.now()}] Loaded {len(ds)} conversations")
        
        # Filter to only English conversations
        print(f"[{datetime.now()}] Filtering to English conversations only...")
        ds = ds.filter(lambda x: x.get("language") == "English")
        print(f"[{datetime.now()}] Filtered to {len(ds)} English conversations")
        print(f"[{datetime.now()}] Memory usage: {memory_usage()}")
        
        # Process conversations
        all_entries = []
        print(f"[{datetime.now()}] Processing conversations...")
        
        for conv_idx, conversation_row in enumerate(tqdm(ds, desc="Processing conversations")):
            try:
                entries = process_conversation(conversation_row)
                all_entries.extend(entries)
                
                # Periodic progress updates
                if (conv_idx + 1) % 10000 == 0:
                    print(f"[{datetime.now()}] Processed {conv_idx + 1} conversations, {len(all_entries)} training examples")
                    print(f"[{datetime.now()}] Memory usage: {memory_usage()}")
            
            except Exception as e:
                print(f"Error processing conversation {conv_idx}: {e}")
                continue
        
        print(f"[{datetime.now()}] Processed {len(ds)} conversations into {len(all_entries)} training examples")
        print(f"[{datetime.now()}] Memory before converting to Dataset: {memory_usage()}")
        
        # Convert to Dataset
        print(f"[{datetime.now()}] Converting to Dataset...")
        dataset = Dataset.from_list(all_entries, features=FEATURES)
        print(f"[{datetime.now()}] Converted to Dataset with {len(dataset)} examples")
        print(f"[{datetime.now()}] Memory after creating Dataset: {memory_usage()}")
        
        # Push to hub if requested
        if self.push_to_hub:
            print(f"[{datetime.now()}] Pushing to Hugging Face Hub: {self.push_to_hub}")
            dataset.push_to_hub(self.push_to_hub, split="train", private=True)
            print(f"[{datetime.now()}] Pushed to hub")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WildChat Dataset Processor")
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push dataset to Hugging Face Hub (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to .env file with necessary credentials",
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to process (for testing)",
    )
    
    args = parser.parse_args()
    
    dataset = WildChatDataset(
        push_to_hub=args.push_to_hub,
        config_path=args.config,
        max_conversations=args.max_conversations,
    )
    
    asyncio.run(dataset.create_raw_dataset())

