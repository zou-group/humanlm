import asyncio
import copy
import json
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import litellm
import polars as pl
from dotenv import load_dotenv


PERSONA_PROMPT_TEMPLATE = """You are an expert at analyzing a {app_name} user behavior. You should generate a JSON object to describe user persona based a target user's responses to some contexts. The contexts ONLY provide other people' posts, and you should NOT use them to infer the target user's demographics. You should ONLY use the target user's responses to summarize the persona.

## Context and Responses:
{comments_text}

## Aspects to cover:

1. Demographics:
- Use explicit subfields: "age group", "gender", "location", "occupation", "nationality", "other"
- Fill with explicit info if available, otherwise "NA".

2. Interests:
- What subjects or themes do they frequently respond on?

3. Values:
- What opinions, attitudes, or worldviews are reflected in their responses?

4. Communication:
- What are their writing styles and formatting habits?

5. Statistics:
- Average / Minimum / Maximum response length (in words). Most frequent words or phrases. Variations in sentence structure and so on.

## Output (strict JSON):
{{
    "analysis": <str>,
    "demographics": {{
        "age group": <str>,
        "gender": <str>,
        "location": <str>,
        "occupation": <str>,
        "nationality": <str>,
        "other": <str>
    }},
    "interests": <a list of 8-12 phrases>,
    "values": <a list of 8-12 phrases>,
    "communication": <a list of 8-12 phrases>,
    "statistics": <a list of 5-10 phrases>
}}

## Instructions:
- [CRITICAL] You MUST always include ALL fields in the JSON output, including "demographics" with ALL its subfields. If demographic information is not explicitly mentioned in the user's responses, set all demographic fields to "NA" but still include them.
- "age group" field: Identify if the user mentioned being X years old in a response from year Y. And find the year of their last response, say Z. Then calculate their age group as (X + (Z - Y)). If no explicit age mentioned, set to "NA".
- "demographics" fields: When extracting demographics, only use explicitly mentioned information. Base your evidence on the user's responses. Do not make assumptions or guesses. If no explicit information is available, use "NA" for each field but ALWAYS include the demographics object.
- [Important!] Other fields: Ensure the phrases are specific, evidence-based, and describe comprehensive aspects of the user. You should quote parts of the user's actual responses as evidence in each phrase without metionining the example index. Avoid vague or generic phrases. Instead, reflect the user's unique traits, behaviors, or preferences.
- "analysis" field: Provide a detailed and step-by-step analysis with the evidence and your reasoning to obtain the user's demongraphics, interests, values, communication style, and statistics.

Your Output:
"""


def extract_json(s):
    def convert_value(value):
        true_values = {"true": True, "false": False, "null": None}
        value_lower = value.lower()
        if value_lower in true_values:
            return true_values[value_lower]
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value  # Return as string if not a number

    def parse_number(s, pos):
        start = pos
        while pos < len(s) and s[pos] in "-+0123456789.eE":
            pos += 1
        num_str = s[start:pos]
        try:
            if "." in num_str or "e" in num_str.lower():
                return float(num_str), pos
            else:
                return int(num_str), pos
        except ValueError:
            raise ValueError(f"Invalid number at position {start}: {num_str}")

    def skip_whitespace(s, pos):
        while pos < len(s) and s[pos] in " \t\n\r":
            pos += 1
        return pos

    def parse_string(s, pos):
        quote_char = s[pos]
        assert quote_char in ('"', "'")
        pos += 1
        result = ""
        while pos < len(s):
            c = s[pos]
            if c == "\\":
                pos += 1
                if pos >= len(s):
                    raise ValueError("Invalid escape sequence")
                c = s[pos]
                escape_sequences = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", quote_char: quote_char}
                result += escape_sequences.get(c, c)
            elif c == quote_char:
                pos += 1
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += c
            pos += 1
        raise ValueError("Unterminated string")

    def parse_key(s, pos):
        pos = skip_whitespace(s, pos)
        if s[pos] in ('"', "'"):
            key, pos = parse_string(s, pos)
            return key, pos
        else:
            raise ValueError(f"Expected string for key at position {pos}")

    def parse_object(s, pos):
        obj = {}
        assert s[pos] == "{"
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "}":
            pos = skip_whitespace(s, pos)
            key, pos = parse_key(s, pos)
            pos = skip_whitespace(s, pos)
            if pos >= len(s) or s[pos] != ":":
                raise ValueError(f'Expected ":" at position {pos}')
            pos += 1
            pos = skip_whitespace(s, pos)
            value, pos = parse_value(s, pos)
            obj[key] = value
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "}":
                break
            elif pos < len(s) and s[pos] != "}":
                raise ValueError(f'Expected "," or "}}" at position {pos}')
        if pos >= len(s) or s[pos] != "}":
            raise ValueError(f'Expected "}}" at position {pos}')
        pos += 1
        return obj, pos

    def parse_array(s, pos):
        lst = []
        assert s[pos] == "["
        pos += 1
        pos = skip_whitespace(s, pos)
        while pos < len(s) and s[pos] != "]":
            value, pos = parse_value(s, pos)
            lst.append(value)
            pos = skip_whitespace(s, pos)
            if pos < len(s) and s[pos] == ",":
                pos += 1
                pos = skip_whitespace(s, pos)
            elif pos < len(s) and s[pos] == "]":
                break
            elif pos < len(s) and s[pos] != "]":
                raise ValueError(f'Expected "," or "]" at position {pos}')
        if pos >= len(s) or s[pos] != "]":
            raise ValueError(f'Expected "]" at position {pos}')
        pos += 1
        return lst, pos

    def parse_triple_quoted_string(s, pos):
        if s[pos : pos + 3] == "'''":
            quote_str = "'''"
        elif s[pos : pos + 3] == '"""':
            quote_str = '"""'
        else:
            raise ValueError(f"Expected triple quotes at position {pos}")
        pos += 3
        result = ""
        while pos < len(s):
            if s[pos : pos + 3] == quote_str:
                pos += 3
                # Attempt to convert to a number if possible
                converted_value = convert_value(result)
                return converted_value, pos
            else:
                result += s[pos]
                pos += 1
        raise ValueError("Unterminated triple-quoted string")

    def parse_value(s, pos):
        pos = skip_whitespace(s, pos)
        if pos >= len(s):
            raise ValueError("Unexpected end of input")
        if s[pos] == "{":
            return parse_object(s, pos)
        elif s[pos] == "[":
            return parse_array(s, pos)
        elif s[pos : pos + 3] in ("'''", '"""'):
            return parse_triple_quoted_string(s, pos)
        elif s[pos] in ('"', "'"):
            return parse_string(s, pos)
        elif s[pos : pos + 4].lower() == "true":
            return True, pos + 4
        elif s[pos : pos + 5].lower() == "false":
            return False, pos + 5
        elif s[pos : pos + 4].lower() == "null":
            return None, pos + 4
        elif s[pos] in "-+0123456789.":
            return parse_number(s, pos)
        else:
            raise ValueError(f"Unexpected character at position {pos}: {s[pos]}")

    json_start = s.index("{")
    json_end = s.rfind("}")
    s = s[json_start : json_end + 1]

    s = s.strip()
    result, pos = parse_value(s, 0)
    pos = skip_whitespace(s, pos)
    if pos != len(s):
        raise ValueError(f"Unexpected content at position {pos}")
    return result


def check_persona(persona: dict) -> bool:
    if not isinstance(persona, dict):
        return False

    # Check top-level keys match exactly
    required_keys = {"demographics", "interests", "values", "communication", "statistics"}
    if set(persona.keys()) != required_keys:
        return False

    # Check demographics structure and content
    demographics = persona["demographics"]
    if not isinstance(demographics, dict):
        return False

    demographics_required_keys = {"age group", "gender", "location", "occupation", "nationality", "other"}
    if set(demographics.keys()) != demographics_required_keys:
        return False

    if not all(isinstance(v, str) for v in demographics.values()):
        return False

    # Check list fields: must be non-empty lists of strings
    list_fields = ["interests", "values", "communication", "statistics"]
    for field in list_fields:
        value = persona[field]
        if not isinstance(value, list) or not value:
            return False
        if not all(isinstance(item, str) for item in value):
            return False

    return True


class UserPersonaGenerator:
    """Minimal user persona generator using litellm"""

    def __init__(self, app_name: str, max_concurrent_users: int = 1, config_path: Optional[str] = None, num_retries=5):
        """Initialize with credentials"""
        self.app_name = app_name
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()
        self.num_retries = num_retries

        if max_concurrent_users <= 1:
            raise ValueError("max_concurrent_users must >= 1")
        self.sem = asyncio.Semaphore(max_concurrent_users)  # To limit concurrent API calls.

    async def generate_persona(self, user_id, history_text, use_user_profile_fields, user_metadata, **llm_kwargs) -> tuple[str, dict]:
        """Generate simple persona using LLM via litellm

        Returns:
            Tuple of (persona_dict, cost_info_dict)
            cost_info contains: prompt_tokens, completion_tokens, total_tokens, cost_usd
        """
        def get_default_persona():
            """Return a default empty persona when generation fails"""
            default_persona = {
                "demographics": {
                    "age group": "NA",
                    "gender": "NA",
                    "location": "NA",
                    "occupation": "NA",
                    "nationality": "NA",
                    "other": "NA"
                },
                "interests": ["Unable to determine interests"],
                "values": ["Unable to determine values"],
                "communication": ["Unable to determine communication style"],
                "statistics": ["Unable to determine statistics"]
            }
            # Update with user metadata if provided
            if use_user_profile_fields is not None and user_metadata is not None:
                default_persona.update(user_metadata)
            return default_persona

        llm_kwargs = copy.deepcopy(llm_kwargs)
        async with self.sem:
            for i in range(self.num_retries):
                try:
                    prompt = PERSONA_PROMPT_TEMPLATE.format(app_name=self.app_name, comments_text=history_text)
                    cost_info = {"cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

                    print(f"{history_text[:1000]}\n\n\n")
                    response_obj = await litellm.acompletion(
                        messages=[{"role": "user", "content": prompt}],
                        **llm_kwargs
                    )
                    response = response_obj.choices[0].message.content

                    # Extract cost info from litellm response
                    if hasattr(response_obj, 'usage') and response_obj.usage:
                        cost_info["prompt_tokens"] = getattr(response_obj.usage, 'prompt_tokens', 0)
                        cost_info["completion_tokens"] = getattr(response_obj.usage, 'completion_tokens', 0)
                        cost_info["total_tokens"] = getattr(response_obj.usage, 'total_tokens', 0)

                    # Get cost from litellm
                    try:
                        cost_info["cost_usd"] = float(litellm.completion_cost(response_obj))
                    except Exception:
                        # Fallback: cost will be 0 if not available
                        cost_info["cost_usd"] = 0.0

                    persona_text = response.strip()
                    persona = extract_json(persona_text)
                    persona.pop("analysis", None)

                    print(f"Generated persona for user {user_id} on attempt {i+1}: \n{persona}")
                    if use_user_profile_fields is not None and user_metadata is not None:
                        persona.update(user_metadata)

                    assert check_persona(persona), f"Invalid persona format: {persona}"
                    return persona, cost_info

                except Exception as e:
                    if i < self.num_retries - 1:
                        time.sleep(60)
                        llm_kwargs['temperature'] = max(llm_kwargs.get('temperature', 0), 0.2)
                        print(f"Error generating persona: {e} for {user_id}. Retrying...")
                    else:
                        print(f"Error generating persona: {e} for {user_id}. Failed after {self.num_retries} retries. Using default empty persona.")
                        # Return default persona instead of raising
                        default_persona = get_default_persona()
                        return default_persona, cost_info

            # If we get here, all retries failed - return default persona
            print(f"Failed to generate persona for {user_id} after {self.num_retries} retries. Using default empty persona.")
            default_persona = get_default_persona()
            return default_persona, {"cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def load_persona_checkpoint(checkpoint_path: Optional[str]) -> Tuple[Optional[pl.DataFrame], set[str]]:
    """
    Load persona checkpoint if it exists.

    Returns:
        Tuple of (checkpoint_df, processed_users_set)
    """
    checkpoint_df = None
    processed_users = set()

    if checkpoint_path is not None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        processed_json_path = os.path.join(checkpoint_dir, "processed_users.json")

        if os.path.exists(checkpoint_path):
            print(f"[{datetime.now()}] Loading persona checkpoint from {checkpoint_path}...")
            checkpoint_df = pl.read_parquet(checkpoint_path)
            processed_users = set(checkpoint_df["user_id"].to_list())
            print(f"[{datetime.now()}] Found {len(processed_users)} users already processed in checkpoint")

        if os.path.exists(processed_json_path):
            with open(processed_json_path, "r") as f:
                processed_users = set(json.load(f))

    return checkpoint_df, processed_users


def save_persona_checkpoint(
    checkpoint_path: str,
    batch_df: pl.DataFrame,
    checkpoint_df: Optional[pl.DataFrame],
    processed_users: set[str],
) -> pl.DataFrame:
    """
    Save persona checkpoint after processing a batch.

    Args:
        checkpoint_path: Path to save checkpoint
        batch_df: DataFrame with new personas to add
        checkpoint_df: Existing checkpoint DataFrame (None if first batch)
        processed_users: Set of processed user IDs

    Returns:
        Updated checkpoint_df
    """
    # Convert persona dict to JSON string for storage
    batch_df = batch_df.with_columns(
        pl.col("persona").map_elements(
            lambda p: json.dumps(p) if isinstance(p, dict) else str(p),
            return_dtype=pl.String
        ).alias("persona")
    )

    # Merge with existing checkpoint
    if checkpoint_df is not None:
        checkpoint_df = pl.concat([checkpoint_df, batch_df], how="vertical")
    else:
        checkpoint_df = batch_df

    # Update processed users set
    processed_users.update(batch_df["user_id"].to_list())

    # Save checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_df.write_parquet(checkpoint_path)
    processed_json_path = os.path.join(checkpoint_dir, "processed_users.json")
    with open(processed_json_path, "w") as f:
        json.dump(sorted(list(processed_users)), f)
    print(f"[{datetime.now()}] Saved checkpoint: {len(processed_users)} users processed")

    return checkpoint_df
