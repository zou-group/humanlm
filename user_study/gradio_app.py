# =============================================================================
# Gradio App for User Study: Evaluating User Simulator Responses
# =============================================================================
# Required environment variables (set in your shell or in a .env file):
#   ANTHROPIC_API_KEY  - Anthropic API key (used by litellm for persona generation)
#   HF_TOKEN           - HuggingFace token (for uploading results to HF datasets)
#
# You can create a .env file in the project root or this directory with these
# variables. By default, python-dotenv loads .env from the current working
# directory. Alternatively, set these variables in your shell environment.
#
# Before running, start the vLLM model servers:
#   vllm serve Qwen/Qwen3-8B --dtype auto --host 0.0.0.0 --port 8000 --tensor-parallel-size 3 --max-model-len 7168
#   vllm serve snap-stanford/standard_grpo_think_opinion --dtype auto --host 0.0.0.0 --port 23457 --tensor-parallel-size 2 --max-model-len 7168
#   vllm serve snap-stanford/humanlm-opinions --dtype auto --host 0.0.0.0 --port 63456 --tensor-parallel-size 2 --max-model-len 7168
# =============================================================================

import argparse
import asyncio
import html
import json
import os
import random
import re
import time
import uuid

from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI
import gradio as gr
import pandas as pd
from huggingface_hub import HfApi
from litellm import completion
from logger_config import logger
from utils import persist, sanitize_filename

load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluating User Simulator Responses')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to skip validation constraints')
args = parser.parse_args()

# DEBUG MODE: Enable with --debug flag to skip validation constraints for testing
DEBUG_MODE = args.debug

# Base directory for resolving relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ENABLE_THINKING = True
base_model_repo = 'Qwen/Qwen3-8B'
GRPO_model_repo = 'snap-stanford/grpo_ablation_reddit_think'
humanlm_model_repo = 'snap-stanford/humanlm_hetero_reddit_best_humanlike'
USER_PROFILE_CACHE_PATH = os.path.join(BASE_DIR, "reddit_user_profile_cache.json")
PERSONA_PROMPT_TEMPLATE_PATH = os.path.join(BASE_DIR, "persona_summarize_prompt.txt")
FOLDER_PATH = os.path.join(BASE_DIR, "annotations")

client_generation_kwargs = {
    "max_completion_tokens": 2048,
    "temperature": 0.0,
    "top_p": 1.0,
}

# Async clients for parallel API calls
base_model_client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    timeout=60.0,
)
grpo_model_client = AsyncOpenAI(
    base_url="http://localhost:23457/v1",
    timeout=60.0,
)
humanlm_model_client = AsyncOpenAI(
    base_url="http://localhost:63456/v1",
    timeout=60.0,
)

# Model client and repo mapping
MODEL_CONFIG = {
    "base": {"client": base_model_client, "repo": base_model_repo},
    "grpo": {"client": grpo_model_client, "repo": GRPO_model_repo},
    "humanlm": {"client": humanlm_model_client, "repo": humanlm_model_repo},
}

# Model-specific settings
ENABLE_THINKING_DICT = {
    "base": ENABLE_THINKING,
    "grpo": ENABLE_THINKING,
    "humanlm": ENABLE_THINKING,
}

MAX_COMPLETION_TOKENS_DICT = {
    key: 256 if ENABLE_THINKING_DICT[key] == False else client_generation_kwargs.get("max_completion_tokens", 1024)
    for key in MODEL_CONFIG.keys()
}

def get_model_client_and_repo(model_type):
    """Get the client and repo for a given model type."""
    config = MODEL_CONFIG.get(model_type, MODEL_CONFIG["base"])
    return config["client"], config["repo"]

# Load persona summarization prompt template
with open(PERSONA_PROMPT_TEMPLATE_PATH, "r") as f:
    PERSONA_PROMPT_TEMPLATE = f.read()

def get_persona_prompt_without_examples(user_profile):
    """
    Generate the prompt for persona summarization without the few-shot examples.
    This is used for logging purposes.
    """
    demographics = user_profile.get("demographics", {})
    values = user_profile.get("values", {})
    communication = user_profile.get("communication", {})
    ranking_raw = values.get("values_ranking_raw", {})

    # Create a condensed prompt with just the input section (no few-shot examples)
    prompt_input_only = f"""Demographics:
- Age Group: {demographics.get("age_group", "NA")}
- Gender: {demographics.get("gender", "NA")}
- Occupation: {demographics.get("occupation", "NA")}
- Location: {demographics.get("location", "NA")}
- Nationality: {demographics.get("nationality", "NA")}

Values Questions:
Q1 (Value Rankings): Freedom: {ranking_raw.get("Freedom", "NA")}, Health: {ranking_raw.get("Health", "NA")}, Wealth: {ranking_raw.get("Wealth", "NA")}, Success: {ranking_raw.get("Success", "NA")}, Happiness: {ranking_raw.get("Happiness", "NA")}
Why #1: {values.get("values_ranking_reason", "")}

Q2 (Handling criticism from family/friends): {values.get("handling_criticism", "")}

Q3 (Forgiveness factors): {values.get("forgiveness_factors", "")}

Q4 (Self vs others' expectations): {values.get("self_vs_others", "")}

Communication Questions:
Q5 (Conflict timing): {communication.get("conflict_timing", "")}

Q6 (Feedback style preference): {communication.get("feedback_style", "")}

Q7 (Supporting friends with problems): {communication.get("supporting_friends", "")}

Q8 (Disagreeing with authority): {communication.get("disagreement_with_authority", "")}"""

    return prompt_input_only


def generate_persona_from_answers(user_profile):
    """
    Generate a summarized persona from user's Q&A responses using claude-haiku-4-5.

    Args:
        user_profile: Dict containing demographics, values, and communication answers

    Returns:
        Tuple of (persona dict, prompt without examples, raw llm response)
    """
    demographics = user_profile.get("demographics", {})
    values = user_profile.get("values", {})
    communication = user_profile.get("communication", {})

    # Format rankings
    ranking_raw = values.get("values_ranking_raw", {})

    # Fill in the prompt template using replace() to avoid issues with JSON curly braces
    prompt = PERSONA_PROMPT_TEMPLATE
    prompt = prompt.replace("{age_group}", demographics.get("age_group", "NA"))
    prompt = prompt.replace("{gender}", demographics.get("gender", "NA"))
    prompt = prompt.replace("{occupation}", demographics.get("occupation", "NA"))
    prompt = prompt.replace("{location}", demographics.get("location", "NA"))
    prompt = prompt.replace("{nationality}", demographics.get("nationality", "NA"))
    prompt = prompt.replace("{rank_freedom}", str(ranking_raw.get("Freedom", "NA")))
    prompt = prompt.replace("{rank_health}", str(ranking_raw.get("Health", "NA")))
    prompt = prompt.replace("{rank_wealth}", str(ranking_raw.get("Wealth", "NA")))
    prompt = prompt.replace("{rank_success}", str(ranking_raw.get("Success", "NA")))
    prompt = prompt.replace("{rank_happiness}", str(ranking_raw.get("Happiness", "NA")))
    prompt = prompt.replace("{q1_answer}", values.get("values_ranking_reason", ""))
    prompt = prompt.replace("{q2_answer}", values.get("handling_criticism", ""))
    prompt = prompt.replace("{q3_answer}", values.get("forgiveness_factors", ""))
    prompt = prompt.replace("{q4_answer}", values.get("self_vs_others", ""))
    prompt = prompt.replace("{q5_answer}", communication.get("conflict_timing", ""))
    prompt = prompt.replace("{q6_answer}", communication.get("feedback_style", ""))
    prompt = prompt.replace("{q7_answer}", communication.get("supporting_friends", ""))
    prompt = prompt.replace("{q8_answer}", communication.get("disagreement_with_authority", ""))

    # Get prompt without few-shot examples for logging
    prompt_for_logging = get_persona_prompt_without_examples(user_profile)

    print("\n" + "="*80)
    print("[PERSONA GENERATION] User Profile Input:")
    print("="*80)
    print(f"Demographics: {json.dumps(demographics, indent=2)}")
    print(f"Values: {json.dumps(values, indent=2)}")
    print(f"Communication: {json.dumps(communication, indent=2)}")
    print("="*80 + "\n")

    # Retry logic for litellm completion
    max_retries = 5
    last_error = None

    for attempt in range(max_retries):
        try:
            print(f"[LiteLLM] Attempting persona generation (attempt {attempt + 1}/{max_retries})...")
            response = completion(
                model="anthropic/claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=0.3,
            )

            response_text = response.choices[0].message.content.strip()

            # Parse JSON from response (handle potential markdown code blocks)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text

            persona = json.loads(json_str)
            print(f"[LiteLLM] Persona generation successful on attempt {attempt + 1}")
            print("\n" + "="*80)
            print("[PERSONA GENERATION] Raw LLM Response:")
            print("="*80)
            print(response_text[:2000] + "..." if len(response_text) > 2000 else response_text)
            print("="*80)
            print("\n[PERSONA GENERATION] Parsed Persona:")
            print("="*80)
            print(json.dumps(persona, indent=2))
            print("="*80 + "\n")
            return persona, prompt_for_logging, response_text

        except Exception as e:
            last_error = e
            print(f"[LiteLLM] Error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                import time
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                print(f"[LiteLLM] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    # All retries failed - create fallback persona with demographics only
    print(f"[LiteLLM] All {max_retries} attempts failed. Last error: {last_error}")
    print(f"[LiteLLM] Using fallback persona with demographics only")

    # Build a simple description based on demographics
    age = demographics.get("age_group", "unknown age")
    gender = demographics.get("gender", "person")
    occupation = demographics.get("occupation", "")
    location = demographics.get("location", "")
    nationality = demographics.get("nationality", "")

    # Create a basic persona description
    demographic_desc = f"A {age} {gender.lower()}"
    if occupation:
        demographic_desc += f" who works as a {occupation}"
    if location or nationality:
        loc_parts = [p for p in [location, nationality] if p]
        demographic_desc += f" from {', '.join(loc_parts)}"

    fallback_persona = {
        "demographics": demographics,
        "values": [
            demographic_desc,
            "Persona details could not be generated due to technical issues"
        ],
        "communication": [
            "Communication style could not be determined due to technical issues"
        ]
    }
    return fallback_persona, prompt_for_logging, f"FALLBACK_PERSONA: Error after {max_retries} retries: {str(last_error)}"


def append_to_hf_csv(hf_api_instance, repo_id, worker_id, user_profile, persona_prompt, persona_output,
                     user_response, model_responses, model_order, evaluations):
    """
    Save submission data as a unique CSV file on HuggingFace to avoid race conditions.
    Each submission gets its own file in the csv_submissions/ folder.

    Args:
        hf_api_instance: HuggingFace API instance
        repo_id: HuggingFace dataset repository ID
        worker_id: Worker/user identifier
        user_profile: Dict containing user's profile from Step 1
        persona_prompt: The prompt sent to litellm (without few-shot examples)
        persona_output: The raw output from litellm
        user_response: User's response to the post (Step 2)
        model_responses: Dict with responses from base, grpo, humanlm models
        model_order: List showing which model was A, B, C
        evaluations: Dict with evaluation results for A, B, C
    """
    import csv
    import tempfile
    import uuid
    from datetime import datetime

    timestamp = datetime.now()

    # Prepare the new row data
    new_row = [
        timestamp.isoformat(),
        worker_id,
        json.dumps(user_profile) if user_profile else "",
        persona_prompt or "",
        persona_output or "",
        user_response or "",
        model_responses.get("base", "") if model_responses else "",
        model_responses.get("grpo", "") if model_responses else "",
        model_responses.get("humanlm", "") if model_responses else "",
        json.dumps(model_order) if model_order else "",
        json.dumps(evaluations.get("evaluation_a", {})) if evaluations else "",
        json.dumps(evaluations.get("evaluation_b", {})) if evaluations else "",
        json.dumps(evaluations.get("evaluation_c", {})) if evaluations else "",
        evaluations.get("humanlikeness_reason", "") if evaluations else "",
        evaluations.get("overall_preference", "") if evaluations else "",
        evaluations.get("additional_feedback", "") if evaluations else ""
    ]

    header = [
        'timestamp',
        'worker_id',
        'user_profile',
        'persona_prompt',
        'persona_output',
        'user_response',
        'base_model_response',
        'grpo_model_response',
        'humanlm_model_response',
        'model_order',
        'evaluation_a',
        'evaluation_b',
        'evaluation_c',
        'humanlikeness_reason',
        'overall_preference',
        'additional_feedback'
    ]

    # Create a unique filename for this submission to avoid race conditions
    unique_id = uuid.uuid4().hex[:8]
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    safe_worker_id = "".join(c if c.isalnum() else "_" for c in str(worker_id))
    csv_filename = f"csv_submissions/{timestamp_str}_{safe_worker_id}_{unique_id}.csv"

    # Create a temporary file with header and the single row
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8') as tmp_file:
        tmp_path = tmp_file.name
        writer = csv.writer(tmp_file)
        writer.writerow(header)
        writer.writerow(new_row)

    # Upload the individual CSV file to HuggingFace
    hf_api_instance.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo=csv_filename,
        repo_id=repo_id,
        repo_type="dataset",
    )

    # Clean up temp file
    os.unlink(tmp_path)


async def completion_with_retry(client, model_repo, prompt, max_retries=3, **kwargs):
    """Call completions API with automatic retry on timeout (async version)."""
    last_error = None
    for attempt in range(max_retries):
        try:
            completion = await client.completions.create(
                model=model_repo,
                prompt=prompt,
                **kwargs
            )
            return completion
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying...")
                await asyncio.sleep(1)  # Brief pause before retry
    raise last_error

# Load the custom chat template for multi-role conversations
# NOTE: Set CHAT_TEMPLATE_PATH to the location of your Jinja chat template file.
CHAT_TEMPLATE_PATH = os.environ.get(
    "CHAT_TEMPLATE_PATH",
    os.path.join(BASE_DIR, "qwen3_multi_role_template_think.jinja"),
)
with open(CHAT_TEMPLATE_PATH, "r") as f:
    CHAT_TEMPLATE = Template(f.read())

def format_prompt(messages, speak_as, persona=None, enable_thinking=False):
    """
    Format messages using the custom jinja template for multi-role conversations.

    Args:
        messages: List of message dicts with 'role', 'content', and optionally 'name'
        speak_as: The name of the speaker who will generate the next response
        persona: Optional persona dict to include as system prompt
        enable_thinking: Whether to enable thinking mode (adds <think> tags)

    Returns:
        Formatted prompt string for the completions API
    """
    # If persona is provided, add system message at the beginning
    if persona:
        system_prompt = get_system_prompt(persona)
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages
    else:
        messages_with_system = messages

    return CHAT_TEMPLATE.render(
        messages=messages_with_system,
        tools=None,
        add_generation_prompt=True,
        speak_as=speak_as,
        enable_thinking=enable_thinking
    )

with open(os.path.join(BASE_DIR, "reddit_post_dict_testset.json"), "r") as f:
    post_id_dict = json.load(f)
post_ids = list(post_id_dict.keys())


# Load system prompt template
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt.txt")
with open(SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT_TEMPLATE = f.read()

def format_persona(persona: dict, field_dropout_prob: float = 0.0, item_dropout_prob: float = 0.0, seed: int = 42) -> str:
    """Parse persona dict to a readable string with deterministic field-level and item-level dropout

    Args:
        persona: Dictionary containing persona information
        field_dropout_prob: Probability used to compute number of fields to drop (0.0 to 1.0)
        item_dropout_prob: Probability of dropping each item within kept fields (0.0 to 1.0)
        seed: Random seed for deterministic dropout
    """
    import hashlib
    if isinstance(persona, str):
        return persona

    # Helper: deterministic hash-based decision
    persona_id = json.dumps(persona, sort_keys=True)

    def hash_decision(key: str, threshold: float) -> bool:
        """Returns True if hash of key is below threshold"""
        if threshold == 0.0:
            return False
        hash_val = int(hashlib.md5(f"{seed}_{persona_id}_{key}".encode()).hexdigest(), 16)
        return (hash_val % 10000) / 10000.0 < threshold

    # Step 1: Field-level dropout - determine which fields to keep
    demographics_has_content = any(v and str(v).strip() != "NA" for k, v in persona.get("demographics", {}).items())

    if field_dropout_prob == 0.0:
        kept_fields = {"demographics", "interests", "values", "communication", "statistics"}
    else:
        # Determine pool of fields that can be dropped
        if demographics_has_content:
            all_fields = ["demographics", "interests", "values", "communication", "statistics"]
            max_drop = 4  # Keep at least 1 field
        else:
            all_fields = ["interests", "values", "communication", "statistics"]
            max_drop = 3  # Keep at least 1 field

        num_drop = min(int(field_dropout_prob * 5), max_drop)
        num_keep = len(all_fields) - num_drop

        # Sort fields by hash and keep top num_keep
        field_hashes = [(int(hashlib.md5(f"{seed}_{persona_id}_field_{f}".encode()).hexdigest(), 16), f)
                        for f in all_fields]
        field_hashes.sort()
        kept_fields = set(f for _, f in field_hashes[:num_keep])

    # Step 2: Build output with item-level dropout on kept fields
    lines = []
    total_items, dropped_items = 0, 0

    # Demographics
    if "demographics" not in kept_fields or not demographics_has_content:
        lines.append("Demographics: Missing")
    else:
        demo_items = []
        for k, v in persona["demographics"].items():
            if v and str(v).strip() != "NA":
                total_items += 1
                if hash_decision(f"item_{k}:{v}", item_dropout_prob):
                    dropped_items += 1
                else:
                    demo_items.append(f"  {k}: {v}")

        if demo_items:
            lines.append("Demographics:")
            lines.extend(demo_items)
        else:
            lines.append("Demographics: Missing")

    # Other aspects - only include fields that exist in the persona
    for aspect in ["interests", "values", "communication", "statistics"]:

        if aspect not in kept_fields:
            lines.append(f"{aspect.capitalize()}: Missing")
        else:
            kept_items = []
            for item in persona.get(aspect, []):
                total_items += 1
                if hash_decision(f"item_{item}", item_dropout_prob):
                    dropped_items += 1
                else:
                    kept_items.append(item)

            if kept_items:
                lines.append(f"{aspect.capitalize()}: \n  " + '\n  '.join(kept_items))
            else:
                lines.append(f"{aspect.capitalize()}: Missing")

    result = "\n".join(lines)
    return result

def get_system_prompt(persona: dict) -> str:
    """Format the system prompt with the given persona."""
    persona_str = format_persona(persona)
    return SYSTEM_PROMPT_TEMPLATE.format(persona=persona_str)

# Runtime state files (created automatically if they don't exist)
_cookies_path = os.path.join(BASE_DIR, "cookies.json")
if os.path.exists(_cookies_path):
    with open(_cookies_path, "r") as f:
        cookies = json.load(f)
else:
    cookies = {}

_worker_id_path = os.path.join(BASE_DIR, "worker_user_id_dict.json")
if os.path.exists(_worker_id_path):
    with open(_worker_id_path, "r") as f:
        worker_user_id_dict = json.load(f)
else:
    worker_user_id_dict = {}
    
# Cache file for user profiles (Step 1 answers) by workerId
if os.path.exists(USER_PROFILE_CACHE_PATH):
    with open(USER_PROFILE_CACHE_PATH, "r") as f:
        user_profile_cache = json.load(f)
else:
    user_profile_cache = {}

def save_user_profile_cache(worker_id, profile_data):
    """Save user profile to cache file. Re-reads the file first to avoid race conditions."""
    global user_profile_cache
    # Re-read the cache file to get the latest data from other users/workers
    if os.path.exists(USER_PROFILE_CACHE_PATH):
        try:
            with open(USER_PROFILE_CACHE_PATH, "r") as f:
                user_profile_cache = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read cache file, using in-memory cache: {e}")

    # Update with the new data
    user_profile_cache[worker_id] = profile_data

    # Write back
    with open(USER_PROFILE_CACHE_PATH, "w") as f:
        json.dump(user_profile_cache, f, indent=4)

def get_cached_user_profile(worker_id):
    """Get cached user profile if it exists."""
    return user_profile_cache.get(worker_id, None)

if not os.path.exists(FOLDER_PATH):
    os.makedirs(FOLDER_PATH)

DATASET_REPO_URL = "snap-stanford/humanlm_reddit_short_mturker_creation"
HF_TOKEN = os.getenv("HF_TOKEN")
print(f"[STARTUP] DATASET_REPO_URL = {DATASET_REPO_URL}")
print(f"[STARTUP] HF_TOKEN is {'SET' if HF_TOKEN else 'NOT SET'}")
if HF_TOKEN:
    hf_api = HfApi(token=HF_TOKEN)
    print(f"[STARTUP] HfApi initialized with token")
else:
    hf_api = HfApi()
    print(f"[STARTUP] WARNING: HfApi initialized WITHOUT token - uploads may fail!")

# Ensure the dataset repository exists, create if it doesn't
def ensure_repo_exists(repo_id, repo_type="dataset"):
    """Create the repository if it doesn't exist."""
    try:
        hf_api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository {repo_id} already exists.")
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            try:
                hf_api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True)
                print(f"Created repository {repo_id}.")
            except Exception as create_error:
                print(f"Warning: Could not create repository {repo_id}: {create_error}")
        else:
            print(f"Warning: Could not check repository {repo_id}: {e}")

# Try to ensure the dataset repo exists at startup
if HF_TOKEN:
    ensure_repo_exists(DATASET_REPO_URL, repo_type="dataset")

latex_delimeter_set = [
    {"left": "\\\\[", "right": "\\\\]", "display": True}, {"left": "\\\\(", "right": "\\\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True}, {"left": "\\(", "right": "\\)", "display": False},
    {"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False},
    {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
    {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
    {"left": "\\begin{align*}", "right": "\\end{align*}", "display": True},
    {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
    {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
    {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
]

# Global dictionary to store persona generation threads (threads can't be serialized in Gradio state)
persona_generation_threads = {}

# Global dictionary to store generated personas (Gradio state is serialized between callbacks,
# so background thread updates to state don't persist)
generated_personas_cache = {}

def load_instance(request: gr.Request):
    query_params = dict(request.query_params)
    username = query_params.get("username", "anonymous")
    folder_path = os.path.join(FOLDER_PATH, username)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # for mturk user set user_id to their previous user_id in case they remove the cookies
    if query_params.get("workerId", "") in worker_user_id_dict:
        user_id = worker_user_id_dict[query_params.get("workerId", "")]
    else:
        if "user_id" in request.cookies and request.cookies["user_id"] != "null":
            user_id = request.cookies["user_id"]
        else:
            user_id = str(uuid.uuid4())
        
    start_time = time.time()

    # All three models are used: base, grpo, humanlm (order randomized in step 2)

    post_prob = [1.0 / len(post_ids) for _ in range(len(post_ids))]
    post_id = random.choices(post_ids, post_prob)[0]

    # Note: We no longer sample a random persona here.
    # The persona is generated from the user's Q&A answers in Step 1.

    # Check for cached user profile
    worker_id = "DEBUG" if DEBUG_MODE else query_params.get("workerId", "")
    cached_profile = get_cached_user_profile(worker_id) if worker_id else None

    state_dict = {
        "username": username,
        "user_id": user_id,
        "post_id": post_id,
        "assignmentId": query_params.get("assignmentId", ""),
        "hitId": query_params.get("hitId", ""),
        "workerId": worker_id,
        "turkSubmitTo": query_params.get("turkSubmitTo", ""),
        "cheat": query_params.get("cheat", ""),
        "user_queries": [],
        "ai_responses": [],
        "background": {},
        "start_time": start_time,
        "periodical_evals": [],
        "cached_profile": cached_profile  # Store cached profile in state
    }
    print(state_dict)
    user_id_dict = {"user_id": user_id}
    if not state_dict["assignmentId"]:
        return state_dict, user_id_dict, \
                gr.update(visible=True), gr.update(visible=False)
    else:
        if state_dict["assignmentId"] == "ASSIGNMENT_ID_NOT_AVAILABLE":
            return state_dict, user_id_dict, \
                gr.update(visible=False), gr.update(visible=False)
        else:
            return state_dict, user_id_dict, \
                gr.update(visible=False), gr.update(visible=True)

tachyon_head = '''<link rel="stylesheet" href="https://unpkg.com/tachyons@4.12.0/css/tachyons.min.css"/>'''

with gr.Blocks(delete_cache=(60, 3600),
                fill_height=True,
                fill_width=True,
               title="Evaluate AIs that Simulate Users",
               head=tachyon_head,
               theme=gr.themes.Monochrome(),
               css="#chatbot, #chatbot_a, #chatbot_b {height: 600px !important;}"
               "#turn-level-rating { height: calc(100vh - 800px) !important; overflow-y: auto !important; overflow-x: hidden !important;}"
               "footer {visibility: hidden;}"
               ".f4-5 {font-size: 1.05rem;}"
               "hr {margin-top: 0.5em; border: none; height: 1.2px; color: #333;  /* old IE */ background-color: #333;  /* Modern Browsers */}"
               ".gap {gap: 6px}"
               "#back-button {color: black; font-size: 1.05rem;}"
               "#forward-button {color: black; font-size: 1.05rem;}"
               "#edit-button {color: black; font-size: 1.05rem;}"
               "#save-button {color: black; font-size: 1.05rem;}"
               "#finish-model-a-button, #finish-model-b-button {color: #e5a400}" # gold 
                "#optional-feedback .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550; color: #e5a400}" # gold
                "#pre-writing-header .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550; color: #73b5f3}"
                "#problem-bank-solved-accordion {margin-top: 10px}"
                "#problem-bank-solved-accordion .svelte-1w6vloh .svelte-1w6vloh {font-size: 1.05rem; font-weight: 550;}"
                "#problem-bank-solved .paginate,  #problem-bank-solved .label {font-size: 1.05rem; color: #000000}"
                "#problem-bank-solved .paginate button.svelte-p5q82i {margin-right: 0.5em; margin-left: 0.5em;}"
                """#add-your-own-button {
                        background-color: #FFEEE6;  /* A soft, natural skin tone */
                        border: none;
                        color: #2B2322;
                        font-weight: 600;
                        }"""
                """.select-document-type-button {
                    font-weight: 600;
                    background-color: #e8e8e8;  /* A soft, natural skin tone */
                    }"""
                """.add-your-own-inputs {
                    padding: 0rem;
                    }
                """
                "#control-bar {width: 60%; margin: 0 auto;}"
                "#pre_writing_window {width: 60%; margin: 0 auto;}"
                ".bullet_point {padding:0 !important; margin:0 !important; height:40px; !important}"
                "#canvas_editable {padding:0 !important; margin:0 !important;}"
                ".annotation_box {padding:0 !important; margin:0 !important;}"
                "#msg-box {padding:0 !important; margin:0 !important;}"
                """.new_added_question {padding:0 !important; 
                    margin:0 !important; 
                    height:40px !important;
                    border: 1.5px solid black !important;
                    border-radius: 6px !important;}
                """
                """#remove-bullet-point-button {
                    height: 40px;
                    border-radius: 8px;
                    background-color: #db2727;
                    border: none;
                    color: white;
                    font-size: 15px;
                    line-height: 1;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 0;
                }"""
                """
                #add-bullet-point-button {
                    height: 80px;
                    border-radius: 8px;
                    background-color: #2e7d32;  /* Dark green color */
                    border: none;
                    color: white;
                    font-size: 15px;
                    line-height: 1;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 0;
                }"""
                "#instance-description {border: 1.5px solid black !important; border-radius: .5rem; padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem;}"
                "#instance-description span {font-size: 1rem !important;}"
                ".centered-column { width: 100% !important; margin-left: auto !important; margin-right: auto !important;}"
                ".scrollable-row {max-height:2000px !important; overflow-y: auto;}"
                ".scrollable-row > .gr-column {flex: 1; display: flex;flex-direction: column;}"
                "#canvas_column { height: 630px; /* Full viewport height */ display: flex; flex-direction: column; scroll-behavior: smooth; overflow-y: auto;}"
                "#canvas_intro {display: flex; align-items: center; justify-content: center; height: 100%;}"
                "#canvas_intro .intro_container {text-align: center; }"
                "#canvas_markdown {text-align: left; padding-top:5%; padding-bottom:5%; padding-left: 10%; padding-right: 10%;}"
                "#canvas_markdown .md {font-size: 15px; !important}"
                "#canvas_intro.hidden {display: none; /* Remove from layout when hidden */}"
                """
                    .canvas_toolbar_button {
                        background-color: white;
                        font-weight: 500;
                        border: 2px solid #d9d9d9;
                        max-width: 70px !important;
                        min-width: 70px !important;
                        padding: 0.2em 0.2em;
                    }
                """
                "#domain_specific_d_type_row {width: 80% !important}"
                "#other_d_type_row {width: 20% !important}"
            ) as demo:
    
    user_id = gr.JSON(visible=False)
    solved_problems = persist(gr.State(value=[]), cookies)
    state = gr.JSON(visible=False)
    post_id = gr.State('')
    def test(state):
        return state['post_id'] if state else ''
    state.change(test, inputs=[state], outputs=[post_id])
    document_history = gr.State([])
    current_document_index = gr.State(-1)
    
    with gr.Column(elem_classes=["mt2"], visible=True) as starting_window:
        post_id_string = "</em>, <em>".join(post_ids)
        post_id_string = f"<em>{post_id_string}</em>"
        gr.HTML(f"""
            <div style="text-align: center; padding: 50px;">
                <h1 style="font-size: 48px; margin-bottom: 20px;">Evaluating User Simulator Responses</h1>
                <p style="font-size: 22px; margin-bottom: 20px;">
                    In this task, you will evaluate responses from three User Simulators.
                    This task is broken into two simple steps to guide you through the experience.
                </p>
                <div style="max-width: 800px; margin: 0 auto; text-align: left;">
                    <ol style="font-size: 20px; line-height: 1.6;">
                        <li>
                            <strong>Step 1: Answer Questions About Yourself</strong><br>
                            You will answer open-ended questions that help us understand your values, perspectives, and communication style. (25+ words per question)
                        </li>
                        <li>
                            <strong>Step 2: Write, Annotate, and Compare</strong><br>
                            <ul style="font-size: 17px; margin-top: 5px;">
                                <li><strong>Part 1</strong>: Read a post and write your response (40+ words)</li>
                                <li><strong>Part 2</strong>: Annotate your response's stance, emotion, belief, value, goal, and communication style (10+ words each)</li>
                                <li><strong>Part 3</strong>: Compare your response with 3 AI-generated responses — describe similarities/differences (50+ words each) and rate similarity and human-likeness</li>
                            </ul>
                        </li>
                    </ol>
                </div>
            </div>
        """)

        def check_both_boxes(checkbox1, checkbox2):
            """
            Return interactive=True only if both checkboxes are checked.
            """
            if checkbox1 and checkbox2:
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)

        with gr.Row():
            # Column for original data collection notice
            with gr.Column(elem_classes=["ba", "pa3"]):
                gr.HTML("""
                    <div class="f4-5">
                    <p><b>Data Collection Notice:</b></p>
                    <p>Before you begin, please note that by checking the box below, you agree to:
                    <ul>
                        <li>Allow us to collect your annotations for research</li>
                        <li>Have your annotations shared publicly as part of our research data</li>
                    </ul>
                    To protect your privacy, please do NOT include any personal identifying information (PII) in your annotations. For instance, don't provide your real name when you write down your information or respond to the post.
                    If you prefer not to participate, you can simply close this window.
                    </p>
                    </div>
                """)
                agreement_check_box1 = gr.Checkbox(
                    label="I agree that my annotations will be collected and shared publicly as research data.",
                    value=False,
                    elem_id="consent-checkbox",
                    interactive=True,
                    elem_classes=["f4-5"]
                )

            # Column for the new note
            with gr.Column(elem_classes=["ba", "pa3"]):
                gr.HTML("""
                    <div class="f4-5">
                    <p><b>Important Note:</b></p>
                    <ul>
                        <li>You should try to provide information that truly reflects you in the real life.</li>
                        <li>Please don't copy-paste from other websites or sources when responding to the post.</li>
                    </p>
                    </div>
                """)
                agreement_check_box2 = gr.Checkbox(
                    label="I have read these notes carefully.",
                    value=False,
                    interactive=True,
                    elem_classes=["f4-5"]
                )

        # Add a button at the bottom
        starting_button = gr.Button(
            value="Let's Start the Task",
            variant="primary",
            size="lg",
            elem_classes=["mt-4"],  # Adds spacing above the button
            interactive=False
        )

        # Use the same function on the change events of both checkboxes
        # The function will check if both are True, then update the button accordingly.
        agreement_check_box1.change(
            fn=check_both_boxes,
            inputs=[agreement_check_box1, agreement_check_box2],
            outputs=starting_button
        )
        agreement_check_box2.change(
            fn=check_both_boxes,
            inputs=[agreement_check_box1, agreement_check_box2],
            outputs=starting_button
        )


    with gr.Column(visible=False, elem_id="pre_writing_window") as pre_writing_window:
        gr.HTML("""
        <div style="text-align: center; padding: 10px;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0">Step 1</h2>
            <p style="font-size: 20px; margin-bottom: 30px;">
                Tell us about yourself by answering these questions.
            </p>
        </div>
        """)

        gr.HTML("""
            <div class="pa3 br3 f6 lh-copy f4-5" style="background-color: #fbe0e0; margin-bottom: 15px;">
                <p style="color: black;">
                    Please answer these quick questions about yourself. Be genuine—there are no right or wrong answers. 
                    <strong style="color: black;">Each answer must be at least 25 words.</strong>
                    Write in complete sentences and explain your reasoning where applicable.
                </p>
                <p style="color: black;">
                Note: If you have done this task before, your previous answers will be pre-filled, but feel free to update them if your perspectives have changed.
                </p>
            </div>
        """)

        # Demographics Section
        gr.HTML("<h3 style='margin-bottom: 10px;'>Basic Information (Required)</h3>")
        with gr.Row():
            with gr.Column(scale=1):
                age_group = gr.Dropdown(
                    choices=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
                    label="Age Group *",
                    value="25-34" if DEBUG_MODE else None
                )
            with gr.Column(scale=1):
                gender = gr.Dropdown(
                    choices=["Male", "Female", "Non-binary", "Other", "Prefer not to say"],
                    label="Gender *",
                    value="Female" if DEBUG_MODE else None
                )
            with gr.Column(scale=1):
                occupation = gr.Textbox(
                    label="Occupation *",
                    placeholder="e.g., Software Engineer",
                    value="Software Engineer at a tech startup" if DEBUG_MODE else ""
                )
        with gr.Row():
            with gr.Column(scale=1):
                location = gr.Textbox(
                    label="Location (City/Country) *",
                    placeholder="e.g., New York, USA",
                    value="San Francisco, California, USA" if DEBUG_MODE else ""
                )
            with gr.Column(scale=1):
                nationality = gr.Textbox(
                    label="Nationality *",
                    placeholder="e.g., American",
                    value="American" if DEBUG_MODE else ""
                )

        # ============ VALUES QUESTIONS (4 questions) ============
        gr.HTML("<h3 style='margin-top: 25px; margin-bottom: 5px;'>Your Values</h3>")
        gr.HTML("<p style='color: #666; font-size: 14px; margin-bottom: 15px;'>Please write at least 25 words for each answer.</p>")

        # Value Q1: Life values ranking with radio buttons
        gr.HTML("""
            <div style="background-color: #e8f4e8; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                <p style="color: black; font-weight: 600; margin: 0;">1. Rank these values from 1 (most important) to 5 (least important):</p>
                <p style="color: #666; font-size: 13px; margin: 5px 0 0 0;">Each rank (1-5) should be used exactly once.</p>
            </div>
        """)
        with gr.Row():
            rank_freedom = gr.Radio(
                choices=["1", "2", "3", "4", "5"],
                label="Freedom",
                value="1" if DEBUG_MODE else None
            )
            rank_health = gr.Radio(
                choices=["1", "2", "3", "4", "5"],
                label="Health",
                value="2" if DEBUG_MODE else None
            )
            rank_wealth = gr.Radio(
                choices=["1", "2", "3", "4", "5"],
                label="Wealth",
                value="5" if DEBUG_MODE else None
            )
            rank_success = gr.Radio(
                choices=["1", "2", "3", "4", "5"],
                label="Success",
                value="4" if DEBUG_MODE else None
            )
            rank_happiness = gr.Radio(
                choices=["1", "2", "3", "4", "5"],
                label="Happiness",
                value="3" if DEBUG_MODE else None
            )
        question1_answer = gr.Textbox(
            label="Explain your reasons (at least 25 words) *",
            placeholder="Explain why your top-ranked values matter most to you...",
            lines=2,
            max_lines=4,
            value="Freedom allows me to make my own choices and live life on my own terms without being controlled by others." if DEBUG_MODE else ""
        )

        # Value Q2: Handling criticism from close ones
        gr.HTML("""
            <div style="background-color: #e8f4e8; padding: 10px; border-radius: 8px; margin-bottom: 5px; margin-top: 10px;">
                <p style="color: black; font-weight: 600; margin: 0;">2. A family member or close friend keeps criticizing your life choices (career, partner, lifestyle). How do you handle it?</p>
            </div>
        """)
        question2_answer = gr.Textbox(
            label="",
            placeholder="Your answer...",
            lines=2,
            max_lines=4,
            value="I try to have an honest conversation first, but if they keep pushing, I set firm boundaries and limit how much I share with them about my life." if DEBUG_MODE else ""
        )

        # Value Q3: Forgiveness and accountability
        gr.HTML("""
            <div style="background-color: #e8f4e8; padding: 10px; border-radius: 8px; margin-bottom: 5px; margin-top: 10px;">
                <p style="color: black; font-weight: 600; margin: 0;">3. Someone close to you (friend, family, or partner) seriously hurts you but apologizes. What factors determine whether you forgive them?</p>
            </div>
        """)
        question3_answer = gr.Textbox(
            label="",
            placeholder="Your answer...",
            lines=2,
            max_lines=4,
            value="Whether they truly understand what they did wrong and show genuine change in their behavior over time, not just empty words." if DEBUG_MODE else ""
        )

        # Value Q4: Self vs others (scenario-based)
        gr.HTML("""
            <div style="background-color: #e8f4e8; padding: 10px; border-radius: 8px; margin-bottom: 5px; margin-top: 10px;">
                <p style="color: black; font-weight: 600; margin: 0;">4. A close friend asks you to help them with something important on a day you had set aside for yourself. How do you respond?</p>
            </div>
        """)
        question4_answer = gr.Textbox(
            label="",
            placeholder="Your answer...",
            lines=2,
            max_lines=4,
            value="I'd probably help them since they're a close friend, but I'd be honest about needing to reschedule my personal time. I'd ask how urgent it is first." if DEBUG_MODE else ""
        )

        # ============ COMMUNICATION QUESTIONS (4 questions) ============
        gr.HTML("<h3 style='margin-top: 25px; margin-bottom: 5px;'>Your Communication Style</h3>")
        gr.HTML("<p style='color: #666; font-size: 14px; margin-bottom: 15px;'>Please write at least 25 words for each answer.</p>")

        # Comm Q5: Conflict timing
        gr.HTML("""
            <div style="background-color: #e8f0f8; padding: 10px; border-radius: 8px; margin-bottom: 5px;">
                <p style="color: black; font-weight: 600; margin: 0;">5. When someone hurts your feelings, do you address it right away or wait? Why?</p>
            </div>
        """)
        question5_answer = gr.Textbox(
            label="",
            placeholder="Your answer...",
            lines=2,
            max_lines=4,
            value="I usually wait a bit to cool down first, maybe a few hours, then address it calmly so I don't say something I regret." if DEBUG_MODE else ""
        )

        # Comm Q6: Feedback style
        gr.HTML("""
            <div style="background-color: #e8f0f8; padding: 10px; border-radius: 8px; margin-bottom: 5px; margin-top: 10px;">
                <p style="color: black; font-weight: 600; margin: 0;">6. Do you prefer direct, blunt feedback or a gentler approach? Why?</p>
            </div>
        """)
        question6_answer = gr.Textbox(
            label="",
            placeholder="Your answer...",
            lines=2,
            max_lines=4,
            value="I prefer direct feedback because it saves time and shows respect. I'd rather know exactly what needs improvement than guess." if DEBUG_MODE else ""
        )

        # Comm Q7: Supporting friends
        gr.HTML("""
            <div style="background-color: #e8f0f8; padding: 10px; border-radius: 8px; margin-bottom: 5px; margin-top: 10px;">
                <p style="color: black; font-weight: 600; margin: 0;">7. How do you usually respond when a friend comes to you with a problem?</p>
            </div>
        """)
        question7_answer = gr.Textbox(
            label="",
            placeholder="Your answer...",
            lines=2,
            max_lines=4,
            value="I listen first and ask what kind of support they need - sometimes people want advice, sometimes they just want someone to listen." if DEBUG_MODE else ""
        )

        # Comm Q8: Disagreement with authority
        gr.HTML("""
            <div style="background-color: #e8f0f8; padding: 10px; border-radius: 8px; margin-bottom: 5px; margin-top: 10px;">
                <p style="color: black; font-weight: 600; margin: 0;">8. How do you express disagreement with someone you respect (like a parent or boss)?</p>
            </div>
        """)
        question8_answer = gr.Textbox(
            label="",
            placeholder="Your answer...",
            lines=2,
            max_lines=4,
            value="I express my viewpoint respectfully but clearly, explaining my reasoning. I believe disagreement can be productive when done professionally." if DEBUG_MODE else ""
        )

        move_to_step2_button = gr.Button(value="Proceed to Step 2",
                                                variant="primary",
                                                size="lg")

    # Step 2: Read Post and Write Response
    with gr.Column(visible=False, elem_id="step2_window") as step2_window:
        gr.HTML("""
        <div style="text-align: center; padding: 10px;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0">Step 2 - Part 1: Write Your Response</h2>
            <p style="font-size: 18px; margin-bottom: 20px; color: #666;">
                In this step, you will write your response to a post, then compare it with AI-generated responses.
            </p>
        </div>
        """)

        gr.HTML("""
            <div class="pa3 br3 f6 lh-copy f4-5" style="background-color: #fbe0e0; margin-bottom: 15px;">
                <p style="color: black;">
                    Please read the Reddit post below carefully and write a thoughtful response as if you were replying to this post.
                    <strong style="color: black;">Your response must be at least 40 words.</strong>
                    Try to provide a genuine response that reflects your personality.
                </p>
            </div>
        """)

        # Post display area
        gr.HTML("<h3 style='margin-bottom: 10px;'>Reddit Post (AITA: Am I the Asshole?)</h3>")
        post_content_html = gr.HTML(
            value="<div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6;'><p style='color: black;'>Loading post...</p></div>",
            elem_id="post_content"
        )

        # Response input area
        gr.HTML("<h3 style='margin-top: 20px; margin-bottom: 10px;'>Your Response</h3>")
        gr.HTML("<p style='color: #666; margin-bottom: 10px;'>Your response must be at least 40 words.</p>")
        user_response_textbox = gr.Textbox(
            label="",
            placeholder="Write your response to the post here...",
            lines=6,
            max_lines=12
        )

        gr.HTML("""
        <p style='color: #666; font-size: 14px; margin-top: 15px; margin-bottom: 10px; text-align: center;'>
            <em>Note: It can take up to 1 minute to load the next page. Please be patient.</em>
        </p>
        """)
        continue_to_reflection_button = gr.Button(value="Continue →", variant="primary", size="lg")

    # ============ STEP 2 ANNOTATE: Annotate Your Own Response ============
    with gr.Column(visible=False, elem_id="step2_annotate_window") as step2_annotate_window:
        gr.HTML("""
        <div style="text-align: center; padding: 10px; background-color: #e8f5e9; border-radius: 8px;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0; color: black;">Step 2 - Part 2: Annotate Your Response</h2>
            <p style="font-size: 18px; margin-bottom: 10px; color: #666;">Describe your response's stance, emotion, belief, value, goal, and communication style</p>
        </div>
        """)

        # Original post display (collapsible)
        with gr.Accordion("Original Post (click to expand)", open=False):
            annotate_post_display = gr.HTML(value="")

        # Your Response display
        gr.HTML("<h3 style='margin-top: 20px; margin-bottom: 10px; color: #2e7d32;'>Your Response:</h3>")
        annotate_user_response_display = gr.HTML(
            value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50; min-height: 100px;'><p>Loading...</p></div>"
        )

        # Describe Your Response's attributes
        gr.HTML("""
        <div style='margin-top: 25px; padding: 15px; background-color: #e8f5e9; border-radius: 8px;'>
            <h3 style='margin: 0 0 10px 0; color: #2e7d32;'>Describe Your Response</h3>
            <p style='color: black; margin: 0;'>Describe your response's stance, emotion, belief, value, goal, and communication style. <strong style='color: black;'>Each field requires at least 10 words.</strong></p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='margin-bottom: 5px;'><strong>Stance</strong> — Your position on the topic. Which side do you take? Do you agree or disagree with the post?</p>")
                self_stance = gr.Textbox(label="", placeholder="Describe your stance (10+ words)...", lines=2)
            with gr.Column():
                gr.HTML("<p style='margin-bottom: 5px;'><strong>Emotion</strong> — How do you feel about this topic? What emotions does your response express?</p>")
                self_emotion = gr.Textbox(label="", placeholder="Describe your emotions (10+ words)...", lines=2)

        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='margin-bottom: 5px;'><strong>Belief</strong> — What general beliefs or principles guide your response? What do you believe to be true?</p>")
                self_belief = gr.Textbox(label="", placeholder="Describe your beliefs (10+ words)...", lines=2)
            with gr.Column():
                gr.HTML("<p style='margin-bottom: 5px;'><strong>Value</strong> — What values are important to you in this situation? What matters most?</p>")
                self_value = gr.Textbox(label="", placeholder="Describe your values (10+ words)...", lines=2)

        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='margin-bottom: 5px;'><strong>Goal</strong> — What are you trying to convey or achieve with your response? What message do you want to send?</p>")
                self_goal = gr.Textbox(label="", placeholder="Describe your goal (10+ words)...", lines=2)
            with gr.Column():
                gr.HTML("<p style='margin-bottom: 5px;'><strong>Communication Style</strong> — How do you express yourself? Are you direct or indirect, formal or casual, empathetic or blunt?</p>")
                self_communication = gr.Textbox(label="", placeholder="Describe your communication style (10+ words)...", lines=2)

        gr.HTML("""
        <p style='color: #666; font-size: 14px; margin-top: 15px; margin-bottom: 10px; text-align: center;'>
            <em>Note: It can take up to 1 minute to load the next page. Please be patient.</em>
        </p>
        """)
        continue_to_compare_button = gr.Button(value="Continue to Compare →", variant="primary", size="lg")

    # ============ STEP 2 PART 2: Compare All Responses (Merged) ============
    with gr.Column(visible=False, elem_id="step2_part2_window") as step2_part2_window:
        gr.HTML("""
        <div style="text-align: center; padding: 10px; background-color: #e3f2fd; border-radius: 8px;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0; color: black;">Step 2 - Part 3: Compare All Responses</h2>
            <p style="font-size: 18px; margin-bottom: 10px; color: #666;">Compare your response with three AI-generated responses</p>
        </div>
        """)

        # General instructions (short)
        gr.HTML("""
            <div style="background-color: #e8f5e9; margin: 15px 0; padding: 15px 20px; border-radius: 8px; border: 2px solid #4caf50;">
                <p style='color: black; margin: 0; font-size: 15px;'>
                    <span style='color: #2e7d32; font-weight: 600;'>In this part, you will:</span><br>
                    (1) Read all responses — Review your response and three AI-generated responses<br>
                    (2) Compare responses — Describe similarities and differences across stance, emotion, belief, value, goal, and communication style, then give similarity scores<br>
                    (3) Confirm and rate human-likeness — Review your scores and rate how human-like each AI response sounds
                </p>
            </div>
        """)

        # Original post (collapsible, hidden by default)
        with gr.Accordion("Original Post (click to expand)", open=False):
            step2_post_display_merged = gr.HTML(value="", elem_id="step2_post_merged")

        # Debug mode labels
        step2_model_debug_labels = gr.HTML(value="", visible=DEBUG_MODE)

        # ==================== PART 3.1: READ ALL RESPONSES ====================
        gr.HTML("""
            <div style="background-color: #fff3e0; margin: 25px 0 15px 0; padding: 12px 20px; border-radius: 8px; border-left: 5px solid #ff9800;">
                <h3 style='margin: 0; color: #e65100; font-size: 20px;'>📖 Part 3.1: Read All Responses</h3>
                <p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>Review your response and all three AI-generated responses below.</p>
            </div>
        """)
        gr.HTML("<h4 style='margin-top: 15px; margin-bottom: 10px; color: #2e7d32;'>Your Response</h4>")
        step2_overview_user_response = gr.HTML(
            value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50;'><p>Loading...</p></div>"
        )

        # Your annotations from Part 2
        with gr.Accordion("Your Annotations (from Part 2)", open=True):
            step2_overview_annotations = gr.HTML(
                value="<div style='background-color: #f5f5f5; padding: 10px; border-radius: 6px;'><p>Loading...</p></div>"
            )

        gr.HTML("<h4 style='margin-top: 20px; margin-bottom: 10px; color: #1565c0;'>AI-Generated Responses</h4>")
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #2980b9;'>Response A</p>")
                step2_overview_response_a = gr.HTML(value="<div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; border: 2px solid #2980b9; min-height: 120px;'><p>Loading...</p></div>")
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #27ae60;'>Response B</p>")
                step2_overview_response_b = gr.HTML(value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #27ae60; min-height: 120px;'><p>Loading...</p></div>")
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #8e44ad;'>Response C</p>")
                step2_overview_response_c = gr.HTML(value="<div style='background-color: #f3e5f5; padding: 15px; border-radius: 8px; border: 2px solid #8e44ad; min-height: 120px;'><p>Loading...</p></div>")

        # ==================== PART 3.2: COMPARE RESPONSES ====================
        gr.HTML("""
            <div style="background-color: #e3f2fd; margin: 30px 0 15px 0; padding: 12px 20px; border-radius: 8px; border-left: 5px solid #2196f3;">
                <h3 style='margin: 0; color: #1565c0; font-size: 20px;'>📝 Part 3.2: Compare Responses</h3>
                <p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>For each AI response, describe similarities/differences with your response and give a similarity score (1-10).</p>
            </div>
        """)

        # ==================== RESPONSE A SECTION ====================
        gr.HTML("<hr style='margin: 20px 0; border: none; border-top: 2px solid #2980b9;'>")

        # Header for Response A
        gr.HTML("""
            <div style="background-color: #e3f2fd; margin: 10px 0; padding: 12px 15px; border-radius: 8px; border-left: 4px solid #2980b9;">
                <p style='color: black; margin: 0; font-size: 16px;'><span style='color: #2980b9; font-weight: 600;'>Compare with Response A</span></p>
            </div>
        """)

        # Row: Your Response | Response A
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Your Response (click to expand/collapse)", open=True):
                    step2_user_response_display_a = gr.HTML(
                        value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50; min-height: 80px;'><p>Loading...</p></div>"
                    )
            with gr.Column(scale=1):
                with gr.Accordion("Response A (click to expand/collapse)", open=True):
                    step2_response_display_a = gr.HTML(
                        value="<div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; border: 2px solid #2980b9; min-height: 80px;'><p>Loading...</p></div>"
                    )

        # Row: Tips | Scoring criteria (ABOVE the comparison box)
        with gr.Row():
            with gr.Column(scale=7):
                gr.HTML("""
                    <div style="background-color: #fff3e0; padding: 12px; border-radius: 6px; margin-top: 5px;">
                        <p style='color: black; margin: 0 0 8px 0; font-size: 13px;'><span style='color: #e65100; font-weight: 600;'>Tip: Consider these aspects when comparing your response with Response A:</span></p>
                        <ul style='color: black; font-size: 12px; margin: 0; padding-left: 18px; line-height: 1.5;'>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Stance</span> — Does Response A take the same position on the topic? Does it agree or disagree with the post in the same way you do?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Emotion</span> — Does Response A express similar emotions about this topic?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Belief</span> — Does Response A reflect the same general beliefs or principles as yours?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Value</span> — Does Response A prioritize the same values that matter to you?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Goal</span> — Is Response A trying to convey or achieve the same message as yours?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Communication Style</span> — Does Response A express itself the same way? Direct or indirect, formal or casual, empathetic or blunt?</li>
                        </ul>
                    </div>
                """)
            with gr.Column(scale=3, min_width=180):
                gr.HTML("""
                    <div style="background-color: #e8eaf6; padding: 10px; border-radius: 6px; margin-top: 5px;">
                        <p style='color: black; margin: 0 0 5px 0; font-size: 13px;'><span style='color: #3f51b5; font-weight: 600;'>Overall Similarity Score (1-10):</span></p>
                        <ul style='color: black; font-size: 12px; margin: 0; padding-left: 12px; line-height: 1.4;'>
                            <li style='color: black;'>1-2: Completely different opinions and expression</li>
                            <li style='color: black;'>3-4: Mostly different with minor overlap</li>
                            <li style='color: black;'>5-6: Somewhat similar - some shared points but notable differences</li>
                            <li style='color: black;'>7-8: Mostly similar with minor differences</li>
                            <li style='color: black;'>9-10: Nearly identical in opinions and expression</li>
                        </ul>
                    </div>
                """)

        # Row: Comparison box | Similarity score
        with gr.Row():
            with gr.Column(scale=7):
                comparison_text_a = gr.Textbox(label="Comparison with Response A", placeholder="Describe similarities and differences, and give reasons why you give the overall similarity score (50+ words)...", lines=4, show_label=False)
            with gr.Column(scale=3, min_width=180):
                gr.HTML("<p style='margin: 0 0 5px 0; font-weight: bold; color: #3f51b5; font-size: 13px; text-align: center;'>Similarity (1-10)</p>")
                overall_similarity_a = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="", interactive=True, show_label=False)

        # ==================== RESPONSE B SECTION ====================
        gr.HTML("<hr style='margin: 20px 0; border: none; border-top: 2px solid #27ae60;'>")

        # Header for Response B
        gr.HTML("""
            <div style="background-color: #e8f5e9; margin: 10px 0; padding: 12px 15px; border-radius: 8px; border-left: 4px solid #27ae60;">
                <p style='color: black; margin: 0; font-size: 16px;'><span style='color: #27ae60; font-weight: 600;'>Compare with Response B</span></p>
            </div>
        """)

        # Row: Your Response | Response B
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Your Response (click to expand/collapse)", open=True):
                    step2_user_response_display_b = gr.HTML(
                        value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50; min-height: 80px;'><p>Loading...</p></div>"
                    )
            with gr.Column(scale=1):
                with gr.Accordion("Response B (click to expand/collapse)", open=True):
                    step2_response_display_b = gr.HTML(
                        value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #27ae60; min-height: 80px;'><p>Loading...</p></div>"
                    )

        # Row: Tips | Scoring criteria (ABOVE the comparison box)
        with gr.Row():
            with gr.Column(scale=7):
                gr.HTML("""
                    <div style="background-color: #fff3e0; padding: 12px; border-radius: 6px; margin-top: 5px;">
                        <p style='color: black; margin: 0 0 8px 0; font-size: 13px;'><span style='color: #e65100; font-weight: 600;'>Tip: Consider these aspects when comparing your response with Response B:</span></p>
                        <ul style='color: black; font-size: 12px; margin: 0; padding-left: 18px; line-height: 1.5;'>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Stance</span> — Does Response B take the same position on the topic? Does it agree or disagree with the post in the same way you do?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Emotion</span> — Does Response B express similar emotions about this topic?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Belief</span> — Does Response B reflect the same general beliefs or principles as yours?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Value</span> — Does Response B prioritize the same values that matter to you?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Goal</span> — Is Response B trying to convey or achieve the same message as yours?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Communication Style</span> — Does Response B express itself the same way? Direct or indirect, formal or casual, empathetic or blunt?</li>
                        </ul>
                    </div>
                """)
            with gr.Column(scale=3, min_width=180):
                gr.HTML("""
                    <div style="background-color: #e8eaf6; padding: 10px; border-radius: 6px; margin-top: 5px;">
                        <p style='color: black; margin: 0 0 5px 0; font-size: 13px;'><span style='color: #3f51b5; font-weight: 600;'>Overall Similarity Score (1-10):</span></p>
                        <ul style='color: black; font-size: 12px; margin: 0; padding-left: 12px; line-height: 1.4;'>
                            <li style='color: black;'>1-2: Completely different opinions and expression</li>
                            <li style='color: black;'>3-4: Mostly different with minor overlap</li>
                            <li style='color: black;'>5-6: Somewhat similar - some shared points but notable differences</li>
                            <li style='color: black;'>7-8: Mostly similar with minor differences</li>
                            <li style='color: black;'>9-10: Nearly identical in opinions and expression</li>
                        </ul>
                    </div>
                """)

        # Row: Comparison box | Similarity score
        with gr.Row():
            with gr.Column(scale=7):
                comparison_text_b = gr.Textbox(label="Comparison with Response B", placeholder="Describe similarities and differences, and give reasons why you give the overall similarity score (50+ words)...", lines=4, show_label=False)
            with gr.Column(scale=3, min_width=180):
                gr.HTML("<p style='margin: 0 0 5px 0; font-weight: bold; color: #3f51b5; font-size: 13px; text-align: center;'>Similarity (1-10)</p>")
                overall_similarity_b = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="", interactive=True, show_label=False)

        # ==================== RESPONSE C SECTION ====================
        gr.HTML("<hr style='margin: 20px 0; border: none; border-top: 2px solid #8e44ad;'>")

        # Header for Response C
        gr.HTML("""
            <div style="background-color: #f3e5f5; margin: 10px 0; padding: 12px 15px; border-radius: 8px; border-left: 4px solid #8e44ad;">
                <p style='color: black; margin: 0; font-size: 16px;'><span style='color: #8e44ad; font-weight: 600;'>Compare with Response C</span></p>
            </div>
        """)

        # Row: Your Response | Response C
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Your Response (click to expand/collapse)", open=True):
                    step2_user_response_display_c = gr.HTML(
                        value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50; min-height: 80px;'><p>Loading...</p></div>"
                    )
            with gr.Column(scale=1):
                with gr.Accordion("Response C (click to expand/collapse)", open=True):
                    step2_response_display_c = gr.HTML(
                        value="<div style='background-color: #f3e5f5; padding: 15px; border-radius: 8px; border: 2px solid #8e44ad; min-height: 80px;'><p>Loading...</p></div>"
                    )

        # Row: Tips | Scoring criteria (ABOVE the comparison box)
        with gr.Row():
            with gr.Column(scale=7):
                gr.HTML("""
                    <div style="background-color: #fff3e0; padding: 12px; border-radius: 6px; margin-top: 5px;">
                        <p style='color: black; margin: 0 0 8px 0; font-size: 13px;'><span style='color: #e65100; font-weight: 600;'>Tip: Consider these aspects when comparing your response with Response C:</span></p>
                        <ul style='color: black; font-size: 12px; margin: 0; padding-left: 18px; line-height: 1.5;'>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Stance</span> — Does Response C take the same position on the topic? Does it agree or disagree with the post in the same way you do?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Emotion</span> — Does Response C express similar emotions about this topic?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Belief</span> — Does Response C reflect the same general beliefs or principles as yours?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Value</span> — Does Response C prioritize the same values that matter to you?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Goal</span> — Is Response C trying to convey or achieve the same message as yours?</li>
                            <li style='color: black;'><span style='font-weight: 600; color: black;'>Communication Style</span> — Does Response C express itself the same way? Direct or indirect, formal or casual, empathetic or blunt?</li>
                        </ul>
                    </div>
                """)
            with gr.Column(scale=3, min_width=180):
                gr.HTML("""
                    <div style="background-color: #e8eaf6; padding: 10px; border-radius: 6px; margin-top: 5px;">
                        <p style='color: black; margin: 0 0 5px 0; font-size: 13px;'><span style='color: #3f51b5; font-weight: 600;'>Overall Similarity Score (1-10):</span></p>
                        <ul style='color: black; font-size: 12px; margin: 0; padding-left: 12px; line-height: 1.4;'>
                            <li style='color: black;'>1-2: Completely different opinions and expression</li>
                            <li style='color: black;'>3-4: Mostly different with minor overlap</li>
                            <li style='color: black;'>5-6: Somewhat similar - some shared points but notable differences</li>
                            <li style='color: black;'>7-8: Mostly similar with minor differences</li>
                            <li style='color: black;'>9-10: Nearly identical in opinions and expression</li>
                        </ul>
                    </div>
                """)

        # Row: Comparison box | Similarity score
        with gr.Row():
            with gr.Column(scale=7):
                comparison_text_c = gr.Textbox(label="Comparison with Response C", placeholder="Describe similarities and differences, and give reasons why you give the overall similarity score (50+ words)...", lines=4, show_label=False)
            with gr.Column(scale=3, min_width=180):
                gr.HTML("<p style='margin: 0 0 5px 0; font-weight: bold; color: #3f51b5; font-size: 13px; text-align: center;'>Similarity (1-10)</p>")
                overall_similarity_c = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="", interactive=True, show_label=False)

        # ==================== PART 3.3: CONFIRM AND RATE HUMAN-LIKENESS ====================
        gr.HTML("""
            <div style="background-color: #ffebee; margin: 30px 0 15px 0; padding: 12px 20px; border-radius: 8px; border-left: 5px solid #c62828;">
                <h3 style='margin: 0; color: #c62828; font-size: 20px;'>✅ Part 3.3: Confirm Scores and Rate Human-likeness</h3>
                <p style='margin: 5px 0 0 0; color: #666; font-size: 14px;'>Review and rank responses by similarity, then rate how human-like each AI response sounds.</p>
            </div>
        """)

        # ==================== RANKING CONFIRMATION ====================
        gr.HTML("""
        <div style='margin-top: 15px; padding: 20px; background-color: #fff8e1; border-radius: 8px; border: 2px solid #ff9800;'>
            <h4 style='margin: 0 0 15px 0; color: #e65100;'>Rank Responses by Similarity</h4>
            <p style='color: black; margin: 0 0 10px 0;'>Review all the similarity scores you gave to these three responses. Which one is the <span style='font-weight: 600; color: black;'>most similar</span>, <span style='font-weight: 600; color: black;'>less similar</span>, and <span style='font-weight: 600; color: black;'>least similar</span> to your response?</p>
            <p style='color: black; margin: 0 0 15px 0;'>Give your reasons in the box below (15+ words), then confirm your scores by selecting the final ranking for each response.</p>
        </div>
        """)

        # Ranking reason textbox
        ranking_reason_textbox = gr.Textbox(
            label="Which one is the most similar, less similar, and least similar to your response? Give your reasons. (15+ words)",
            placeholder="Explain your reasoning for the rankings. ",
            lines=3
        )

        # Ranking dropdowns
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #2980b9; margin-bottom: 5px;'>Response A Rank</p>")
                rank_a = gr.Radio(choices=["1st (Most Similar)", "2nd", "3rd (Least Similar)"], label="", value=None)
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #27ae60; margin-bottom: 5px;'>Response B Rank</p>")
                rank_b = gr.Radio(choices=["1st (Most Similar)", "2nd", "3rd (Least Similar)"], label="", value=None)
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #8e44ad; margin-bottom: 5px;'>Response C Rank</p>")
                rank_c = gr.Radio(choices=["1st (Most Similar)", "2nd", "3rd (Least Similar)"], label="", value=None)

        # ==================== HUMANLIKENESS SECTION ====================
        gr.HTML("""
            <div style="background-color: #fce4ec; margin: 20px 0 15px 0; padding: 20px; border-radius: 8px; border: 2px solid #c2185b;">
                <h3 style='margin: 0 0 10px 0; color: #c2185b; font-size: 18px;'>Rate Human-likeness</h3>
                <p style='color: black; margin: 0 0 12px 0; font-size: 14px;'>How human-like does each response sound? Consider whether it reads naturally and could have been written by a real person.</p>
                <p style='color: black; margin: 0 0 8px 0; font-size: 13px;'><span style='color: #c2185b; font-weight: 600;'>Human-likeness Score (1-10):</span></p>
                <ul style='color: black; font-size: 12px; margin: 0; padding-left: 20px; line-height: 1.5;'>
                    <li style='color: black;'>1-2: Very robotic/artificial</li>
                    <li style='color: black;'>3-4: Somewhat unnatural</li>
                    <li style='color: black;'>5-6: Moderately human-like</li>
                    <li style='color: black;'>7-8: Quite natural</li>
                    <li style='color: black;'>9-10: Indistinguishable from human</li>
                </ul>
            </div>
        """)

        # Humanlikeness scores row
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #2980b9; margin-bottom: 5px;'>Response A</p>")
                humanlikeness_score_a = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Human-likeness (1-10)", interactive=True)
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #27ae60; margin-bottom: 5px;'>Response B</p>")
                humanlikeness_score_b = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Human-likeness (1-10)", interactive=True)
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #8e44ad; margin-bottom: 5px;'>Response C</p>")
                humanlikeness_score_c = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Human-likeness (1-10)", interactive=True)

        # Humanlikeness justification
        humanlikeness_reason_textbox = gr.Textbox(
            label="Why did you give these human-likeness scores? (10+ words)",
            placeholder="Explain what made each response sound more or less human-like. ",
            lines=3
        )

        with gr.Row():
            continue_to_review_button = gr.Button(value="Continue to Review →", variant="primary", size="lg")

    # ============ STEP 2 REVIEW: Review All Responses and Rankings ============
    with gr.Column(visible=False, elem_id="step2_review_window") as step2_review_window:
        gr.HTML("""
        <div style="text-align: center; padding: 10px; background-color: #fff3e0; border-radius: 8px;">
            <h2 style="font-size: 32px; margin-bottom: 10px; margin-top: 0; color: black;">Step 2 - Part 4: Review and Confirm</h2>
            <p style="font-size: 18px; margin-bottom: 10px; color: #666;">Review your comparisons and scores, then rank the responses by similarity</p>
        </div>
        """)

        # Original post (collapsible)
        with gr.Accordion("Original Post (click to expand)", open=False):
            step2_review_post_display = gr.HTML(value="")

        # Row 1: Your Response
        gr.HTML("<h3 style='margin-top: 20px; margin-bottom: 10px; color: #2e7d32;'>Your Response</h3>")
        step2_review_user_response = gr.HTML(
            value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50;'><p>Loading...</p></div>"
        )

        # Row 2: Response A | Response B | Response C
        gr.HTML("<h3 style='margin-top: 20px; margin-bottom: 10px;'>AI-Generated Responses</h3>")
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #2980b9;'>Response A</p>")
                step2_review_response_a = gr.HTML(value="<div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; border: 2px solid #2980b9; min-height: 120px;'><p>Loading...</p></div>")
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #27ae60;'>Response B</p>")
                step2_review_response_b = gr.HTML(value="<div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #27ae60; min-height: 120px;'><p>Loading...</p></div>")
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #8e44ad;'>Response C</p>")
                step2_review_response_c = gr.HTML(value="<div style='background-color: #f3e5f5; padding: 15px; border-radius: 8px; border: 2px solid #8e44ad; min-height: 120px;'><p>Loading...</p></div>")

        # Row 3: Comparison texts
        gr.HTML("<h3 style='margin-top: 20px; margin-bottom: 10px;'>Your Comparisons</h3>")
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #2980b9;'>Comparison A</p>")
                step2_review_comparison_a = gr.HTML(value="<div style='background-color: #f5f5f5; padding: 10px; border-radius: 8px; min-height: 80px;'><p>Loading...</p></div>")
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #27ae60;'>Comparison B</p>")
                step2_review_comparison_b = gr.HTML(value="<div style='background-color: #f5f5f5; padding: 10px; border-radius: 8px; min-height: 80px;'><p>Loading...</p></div>")
            with gr.Column():
                gr.HTML("<p style='text-align: center; font-weight: bold; color: #8e44ad;'>Comparison C</p>")
                step2_review_comparison_c = gr.HTML(value="<div style='background-color: #f5f5f5; padding: 10px; border-radius: 8px; min-height: 80px;'><p>Loading...</p></div>")

        # Row 4.1: Human-likeness scores
        gr.HTML("<h3 style='margin-top: 20px; margin-bottom: 10px;'>Human-likeness Scores</h3>")
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='text-align: center; color: #2980b9;'>Response A</p>")
                review_humanlikeness_a = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Human-likeness A (1-10)", interactive=True)
            with gr.Column():
                gr.HTML("<p style='text-align: center; color: #27ae60;'>Response B</p>")
                review_humanlikeness_b = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Human-likeness B (1-10)", interactive=True)
            with gr.Column():
                gr.HTML("<p style='text-align: center; color: #8e44ad;'>Response C</p>")
                review_humanlikeness_c = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Human-likeness C (1-10)", interactive=True)

        # Row 4.2: Overall similarity scores
        gr.HTML("<h3 style='margin-top: 20px; margin-bottom: 10px;'>Overall Similarity Scores</h3>")
        with gr.Row():
            with gr.Column():
                gr.HTML("<p style='text-align: center; color: #2980b9;'>Response A</p>")
                review_similarity_a = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Similarity A (1-10)", interactive=True)
            with gr.Column():
                gr.HTML("<p style='text-align: center; color: #27ae60;'>Response B</p>")
                review_similarity_b = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Similarity B (1-10)", interactive=True)
            with gr.Column():
                gr.HTML("<p style='text-align: center; color: #8e44ad;'>Response C</p>")
                review_similarity_c = gr.Number(minimum=0 if DEBUG_MODE else 1, maximum=10, value=None, precision=0, label="Similarity C (1-10)", interactive=True)

        # Additional feedback
        gr.HTML("<h3 style='margin-top: 30px; margin-bottom: 10px; border-top: 2px solid #ddd; padding-top: 20px;'>Additional Feedback (Optional)</h3>")
        additional_feedback_textbox = gr.Textbox(
            lines=4,
            label="Any other comments about the user simulators and this study?",
            placeholder="Share your thoughts here...",
            interactive=True
        )

        with gr.Row():
            back_to_part2_button = gr.Button(value="← Back to Compare", variant="secondary", size="lg")
            proceed_to_step3_button = gr.Button(value="Finish and Submit →", variant="primary", size="lg")

    # Define skip_to_pre_writing after pre_writing_window is created
    def skip_to_pre_writing(state):
        """Go directly to pre-writing window with open-ended questions.
        If cached profile exists, pre-fill the form fields.
        """
        cached = state.get("cached_profile", None) if state else None

        if cached:
            # Extract cached values
            demo = cached.get("demographics", {})
            vals = cached.get("values", {})
            comm = cached.get("communication", {})
            ranking_raw = vals.get("values_ranking_raw", {})

            return (
                state,
                gr.update(visible=False),  # starting_window
                gr.update(visible=True),   # pre_writing_window
                # Demographics
                gr.update(value=demo.get("age_group", None)),  # age_group
                gr.update(value=demo.get("gender", None)),  # gender
                gr.update(value=demo.get("occupation", "")),  # occupation
                gr.update(value=demo.get("location", "")),  # location
                gr.update(value=demo.get("nationality", "")),  # nationality
                # Value ranking radio buttons
                gr.update(value=ranking_raw.get("Freedom", None)),  # rank_freedom
                gr.update(value=ranking_raw.get("Health", None)),  # rank_health
                gr.update(value=ranking_raw.get("Wealth", None)),  # rank_wealth
                gr.update(value=ranking_raw.get("Success", None)),  # rank_success
                gr.update(value=ranking_raw.get("Happiness", None)),  # rank_happiness
                # Values questions
                gr.update(value=vals.get("values_ranking_reason", "")),  # question1_answer (reason)
                gr.update(value=vals.get("handling_criticism", "")),  # question2
                gr.update(value=vals.get("forgiveness_factors", "")),  # question3
                gr.update(value=vals.get("self_vs_others", "")),  # question4
                # Communication questions
                gr.update(value=comm.get("conflict_timing", "")),  # question5
                gr.update(value=comm.get("feedback_style", "")),  # question6
                gr.update(value=comm.get("supporting_friends", "")),  # question7
                gr.update(value=comm.get("disagreement_with_authority", "")),  # question8
            )
        else:
            # No cached data, return empty updates
            return (
                state,
                gr.update(visible=False),  # starting_window
                gr.update(visible=True),   # pre_writing_window
                gr.update(),  # age_group
                gr.update(),  # gender
                gr.update(),  # occupation
                gr.update(),  # location
                gr.update(),  # nationality
                gr.update(),  # rank_freedom
                gr.update(),  # rank_health
                gr.update(),  # rank_wealth
                gr.update(),  # rank_success
                gr.update(),  # rank_happiness
                gr.update(),  # question1_answer (reason)
                gr.update(),  # question2
                gr.update(),  # question3
                gr.update(),  # question4
                gr.update(),  # question5
                gr.update(),  # question6
                gr.update(),  # question7
                gr.update(),  # question8
            )

    starting_button.click(
        fn=skip_to_pre_writing,
        inputs=[state],
        outputs=[state, starting_window, pre_writing_window,
                 age_group, gender, occupation, location, nationality,
                 rank_freedom, rank_health, rank_wealth, rank_success, rank_happiness,
                 question1_answer, question2_answer, question3_answer, question4_answer,
                 question5_answer, question6_answer, question7_answer, question8_answer]
    )

    # Final submission window - shown after evaluation
    with gr.Column(visible=False) as final_button_col:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h2 style="font-size: 28px; margin-bottom: 10px;">Thank You!</h2>
            <p style="font-size: 18px; margin-bottom: 20px;">
                Your evaluation has been recorded. Click the button below to submit your response.
            </p>
        </div>
        """)
        with gr.Row():
            submit_hit_button = gr.Button(value="Submit the Hit", visible=False, interactive=True)
            submit_hf_button = gr.Button(value="Submit the Task", visible=False, interactive=True)

    def move_to_step2_button_click(state, age_grp, gndr, occ, loc, nat,
                                          r_freedom, r_health, r_wealth, r_success, r_happiness, q1_reason,
                                          q2, q3, q4, q5, q6, q7, q8):
        # Define minimum word count for each answer
        min_words = 25

        if not DEBUG_MODE and not state.get("cheat", ""):
            # Validate mandatory demographics fields
            if not age_grp:
                raise gr.Error("Please select your age group.", duration=5)
            if not gndr:
                raise gr.Error("Please select your gender.", duration=5)

            # Validate ranking radio buttons
            rankings = [r_freedom, r_health, r_wealth, r_success, r_happiness]
            if any(r is None for r in rankings):
                raise gr.Error("Please assign a rank (1-5) to each value.", duration=5)
            if len(set(rankings)) != 5:
                raise gr.Error("Each rank (1-5) should be used exactly once. Please check for duplicates.", duration=5)

        # Fields that only need to be non-empty (no word count requirement)
        required_fields = [
            ("Occupation", occ),
            ("Location", loc),
            ("Nationality", nat),
        ]

        # Fields that need minimum word count
        word_count_fields = [
            ("Values ranking reason", q1_reason),
            ("Question 2", q2),
            ("Question 3", q3),
            ("Question 4", q4),
            ("Question 5", q5),
            ("Question 6", q6),
            ("Question 7", q7),
            ("Question 8", q8),
        ]

        if not DEBUG_MODE and not state.get("cheat", ""):
            # Validate required fields are not empty
            for field_name, answer in required_fields:
                if not answer or not answer.strip():
                    raise gr.Error(f"Please fill in {field_name}.", duration=5)

            # Validate that each question field has at least min_words words
            for field_name, answer in word_count_fields:
                word_count = len(answer.split()) if answer and answer.strip() else 0
                if word_count < min_words:
                    raise gr.Error(f"Please write at least {min_words} words for {field_name}. Current: {word_count} words.", duration=5)

        # Store user profile in state with open-ended responses
        # Convert rankings to a sorted list by rank
        ranking_dict = {
            "Freedom": r_freedom,
            "Health": r_health,
            "Wealth": r_wealth,
            "Success": r_success,
            "Happiness": r_happiness
        }
        sorted_ranking = sorted(ranking_dict.items(), key=lambda x: x[1] if x[1] else "9")
        values_ranking_list = [item[0] for item in sorted_ranking]

        state["user_profile"] = {
            "demographics": {
                "age_group": age_grp,
                "gender": gndr,
                "occupation": occ,
                "location": loc,
                "nationality": nat,
            },
            "values": {
                "values_ranking": values_ranking_list,
                "values_ranking_raw": ranking_dict,
                "values_ranking_reason": q1_reason,
                "handling_criticism": q2,
                "forgiveness_factors": q3,
                "self_vs_others": q4,
            },
            "communication": {
                "conflict_timing": q5,
                "feedback_style": q6,
                "supporting_friends": q7,
                "disagreement_with_authority": q8,
            }
        }

        print("\n" + "="*80)
        print("[STEP 1 -> STEP 2] User Profile (Survey Answers):")
        print("="*80)
        print(json.dumps(state["user_profile"], indent=2))
        print("="*80 + "\n")

        # Generate summarized persona from Q&A using LLM in the background
        # This doesn't block the user from proceeding to Step 2
        # Also pre-generate the 3 model responses so comparison is instant
        def background_persona_and_responses_generation(user_profile, cache_key, post_id):
            """Generate persona and model responses in background and store in global cache."""

            async def async_generate_responses(generated_persona, post_content, poster_name):
                """Async function to generate all model responses in parallel."""
                model_types = ["base", "grpo", "humanlm"]

                async def gen_response(model_type):
                    initial_chat_history = [
                        {"role": "user", "content": post_content, "name": poster_name},
                    ]
                    # Get model-specific settings
                    use_thinking = ENABLE_THINKING_DICT.get(model_type, True)
                    max_tokens = MAX_COMPLETION_TOKENS_DICT.get(model_type, 512)
                    formatted_prompt = format_prompt(
                        messages=initial_chat_history.copy(),
                        speak_as="HUMAN",
                        persona=generated_persona,
                        enable_thinking=use_thinking,
                    )
                    client, model_repo = get_model_client_and_repo(model_type)
                    print(f"\n" + "="*80)
                    print(f"[MODEL SIMULATOR PROMPT] {model_type.upper()} (thinking={use_thinking}, max_tokens={max_tokens})")
                    print("="*80)
                    print(formatted_prompt)
                    print("="*80 + "\n")
                    print(f"[Background] Model ({model_type}) generating response...")

                    try:
                        completion = await completion_with_retry(
                            client, model_repo, formatted_prompt,
                            max_tokens=max_tokens,
                            temperature=client_generation_kwargs.get("temperature", 0.4),
                            top_p=client_generation_kwargs.get("top_p", 0.9),
                            stop=["<|im_end|>", "<|im_start|>"]
                        )
                        raw_reply = completion.choices[0].text.strip()
                        print(f"[Background] Model ({model_type}) raw response: {raw_reply[:100]}...")

                        response = raw_reply
                        if "</think>" in response:
                            response = response.split("</think>", 1)[1].strip()
                        response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
                        if response_match:
                            response = response_match.group(1).strip()
                        response = response.replace("<response>", "").replace("</response>", "").strip()
                        if not response:
                            response = "(No response generated)"
                        print(f"\n" + "-"*60)
                        print(f"[MODEL RESPONSE] {model_type.upper()}")
                        print("-"*60)
                        print(f"Raw: {raw_reply[:500]}..." if len(raw_reply) > 500 else f"Raw: {raw_reply}")
                        print(f"\nParsed: {response}")
                        print("-"*60 + "\n")
                        return model_type, response
                    except Exception as e:
                        print(f"[Background] Error generating response for {model_type}: {e}")
                        return model_type, "(Error generating response)"

                # Run all three model calls in parallel using asyncio.gather
                results = await asyncio.gather(*[gen_response(mt) for mt in model_types])
                return {model_type: response for model_type, response in results}

            print(f"[Background] Thread started! cache_key={cache_key}, post_id={post_id}")
            try:
                # Step 1: Generate persona
                print(f"[Background] Starting persona generation...")
                generated_persona, persona_prompt, persona_raw_output = generate_persona_from_answers(user_profile)
                print(f"[Background] Generated persona: {json.dumps(generated_persona, indent=2)}")

                # Store persona in global cache immediately
                generated_personas_cache[cache_key] = {
                    "generated_persona": generated_persona,
                    "persona_prompt": persona_prompt,
                    "persona_raw_output": persona_raw_output,
                    "responses": {},  # Will be filled below
                    "responses_ready": False
                }

                # Step 2: Generate 3 model responses in parallel using async
                print(f"[Background] Starting model response generation (async)...")
                post_data = post_id_dict.get(post_id, {"post": "No content available", "poster_name": "Unknown"})
                post_content = post_data["post"]
                poster_name = post_data["poster_name"]

                # Store a sample simulator prompt for logging (using base model settings)
                sample_messages = [{"role": "user", "content": post_content, "name": poster_name}]
                sample_simulator_prompt = format_prompt(
                    messages=sample_messages,
                    speak_as="HUMAN",
                    persona=generated_persona,
                    enable_thinking=ENABLE_THINKING_DICT.get("base", True),
                )
                generated_personas_cache[cache_key]["simulator_prompt"] = sample_simulator_prompt

                # Run async function in this thread
                responses = asyncio.run(async_generate_responses(generated_persona, post_content, poster_name))

                # Update cache with responses
                generated_personas_cache[cache_key]["responses"] = responses
                generated_personas_cache[cache_key]["responses_ready"] = True
                print("\n" + "="*80)
                print("[BACKGROUND] All model responses generated and cached!")
                print("="*80)
                for model_name, resp in responses.items():
                    print(f"\n[{model_name.upper()}]: {resp[:300]}..." if len(resp) > 300 else f"\n[{model_name.upper()}]: {resp}")
                print("="*80 + "\n")

            except Exception as e:
                import traceback
                print(f"[Background] Error in background generation: {e}")
                print(f"[Background] Traceback: {traceback.format_exc()}")
                generated_personas_cache[cache_key] = {
                    "generated_persona": {"error": str(e)},
                    "persona_prompt": get_persona_prompt_without_examples(user_profile),
                    "persona_raw_output": f"Error: {str(e)}",
                    "responses": {},
                    "responses_ready": False
                }

        # Start background thread for persona and response generation
        import threading
        user_id = state.get("user_id", str(uuid.uuid4()))
        post_id = state.get("post_id", "")
        # Use user_id + post_id as cache key to ensure unique persona per user-post combination
        thread_key = f"{user_id}_{post_id}"
        persona_thread = threading.Thread(
            target=background_persona_and_responses_generation,
            args=(state["user_profile"].copy(), thread_key, post_id)
        )
        persona_thread.daemon = True
        persona_thread.start()
        # Store thread in global dict (can't serialize thread objects in Gradio state)
        persona_generation_threads[thread_key] = persona_thread
        state["persona_thread_key"] = thread_key
        print(f"Started background persona and response generation (thread key: {thread_key})...")

        # Cache user profile for returning workers
        worker_id = state.get("workerId", "")
        if worker_id:
            save_user_profile_cache(worker_id, state["user_profile"])

        # Get the post content and poster name for this user
        post_id = state.get("post_id", "")
        post_data = post_id_dict.get(post_id, {"post": "No content available", "poster_name": "Unknown"})
        post_content = post_data["post"]
        poster_name = post_data["poster_name"]

        # Store poster_name in state for later use
        state["poster_name"] = poster_name

        # Escape HTML special characters to prevent injection, but preserve the text
        # Then convert newlines to <br> for proper display
        escaped_content = html.escape(post_content).replace('\n', '<br>')

        # Format the post HTML
        post_html = f"""
        <div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6;'>
            <p style='color: black; line-height: 1.6;'>{escaped_content}</p>
        </div>
        """

        # Record the start time for Step 2
        state["step_2_start_time"] = time.time()
        return state, gr.update(visible=False), gr.update(visible=True), post_html

    async def generate_model_response(post_content, poster_name, simulator_persona, model_type):
        """Helper function to generate a single model response (async)."""
        initial_chat_history = [
            {"role": "user", "content": post_content, "name": poster_name},
        ]

        # Get model-specific settings
        use_thinking = ENABLE_THINKING_DICT.get(model_type, True)
        max_tokens = MAX_COMPLETION_TOKENS_DICT.get(model_type, 512)

        formatted_prompt = format_prompt(
            messages=initial_chat_history.copy(),
            speak_as="HUMAN",
            persona=simulator_persona,
            enable_thinking=use_thinking,
        )

        client, model_repo = get_model_client_and_repo(model_type)

        print(f"Model ({model_type}) prompt (thinking={use_thinking}, max_tokens={max_tokens}):", formatted_prompt)
        completion = await completion_with_retry(
            client, model_repo, formatted_prompt,
            max_tokens=max_tokens,
            temperature=client_generation_kwargs.get("temperature", 0.4),
            top_p=client_generation_kwargs.get("top_p", 0.9),
            stop=["<|im_end|>", "<|im_start|>"]
        )
        raw_reply = completion.choices[0].text.strip()
        print(f"Model ({model_type}) raw response:", raw_reply)

        response = raw_reply
        if "</think>" in response:
            response = response.split("</think>", 1)[1].strip()
        response_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
        if response_match:
            response = response_match.group(1).strip()
        # Remove any remaining <response> or </response> tags
        response = response.replace("<response>", "").replace("</response>", "").strip()
        if not response:
            response = "(No response generated)"

        return response

    def continue_to_annotate_click(state, user_response):
        """Transition from Step 2 Part 1 to Annotate Your Response."""
        print("\n" + "="*80)
        print("[CONTINUE TO ANNOTATE] User clicked Continue button")
        print("="*80)

        # Validate word count
        word_count = len(user_response.split()) if user_response.strip() else 0

        if not DEBUG_MODE and not state.get("cheat", ""):
            if word_count < 40:
                raise gr.Error("Please write at least 40 words in your response.", duration=5)

        # Store the user's response
        state["user_post_response"] = user_response

        print(f"User response ({word_count} words):")
        print("-"*60)
        print(user_response)
        print("-"*60 + "\n")

        # Format user response HTML for display in annotation window
        post_id = state.get("post_id", "")
        post_data = post_id_dict.get(post_id, {"post": "No content available", "poster_name": "Unknown"})
        post_content = post_data["post"]
        poster_name = post_data["poster_name"]

        escaped_post = html.escape(post_content).replace('\n', '<br>')
        post_html = f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>
            <p style='color: black; line-height: 1.6;'>{escaped_post}</p>
        </div>
        """
        state["post_html"] = post_html

        escaped_user_response = html.escape(user_response).replace('\n', '<br>')
        user_response_html = f"""
        <div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #4caf50; min-height: 150px;'>
            <p style='color: black; line-height: 1.6;'>{escaped_user_response}</p>
        </div>
        """
        state["user_response_html"] = user_response_html

        return (state, gr.update(visible=False), gr.update(visible=True),
                gr.update(value=post_html),
                gr.update(value=user_response_html))

    def continue_to_compare_click(state, stance, emotion, belief, value, goal, comm):
        """Transition from Annotate to Compare All Responses."""
        print("\n" + "="*80)
        print("[CONTINUE TO COMPARE] User clicked Continue to Compare button")
        print("="*80)

        # Validate annotations (10 words each)
        if not DEBUG_MODE and not state.get("cheat", ""):
            fields = {"Stance": stance, "Emotion": emotion, "Belief": belief, "Value": value, "Goal": goal, "Communication Style": comm}
            for name, val in fields.items():
                word_count = len(val.split()) if val and val.strip() else 0
                if word_count < 10:
                    raise gr.Error(f"Please write at least 10 words for {name}. Current: {word_count} words.", duration=5)

        # Store user's self-annotations
        state["self_description"] = {
            "stance": stance, "emotion": emotion, "belief": belief,
            "value": value, "goal": goal, "communication_style": comm
        }

        print("User Self-Annotations:")
        print("-"*60)
        for name, val in state["self_description"].items():
            print(f"  {name}: {val}")
        print("-"*60 + "\n")

        # Get post content
        post_id = state.get("post_id", "")
        post_data = post_id_dict.get(post_id, {"post": "No content available", "poster_name": "Unknown"})
        post_content = post_data["post"]
        poster_name = post_data["poster_name"]

        # Wait for background model response generation to complete
        thread_key = state.get("persona_thread_key")
        if thread_key and thread_key in persona_generation_threads:
            persona_thread = persona_generation_threads[thread_key]
            if persona_thread.is_alive():
                print("Waiting for background persona and response generation to complete...")
                persona_thread.join(timeout=60)
                if persona_thread.is_alive():
                    print("Warning: Background generation timed out")
            del persona_generation_threads[thread_key]

        # Retrieve model responses from cache
        cached_persona_data = generated_personas_cache.get(thread_key, {}) if thread_key else {}
        responses = cached_persona_data.get("responses", {})
        responses_ready = cached_persona_data.get("responses_ready", False)

        # Fallback if responses not ready
        if not responses_ready or not responses:
            print("[COMPARE] Responses not ready, generating fallback...")
            simulator_persona = cached_persona_data.get("generated_persona", {})
            if not simulator_persona and state.get("user_profile"):
                simulator_persona, _, _ = generate_persona_from_answers(state["user_profile"])

            async def generate_all_responses():
                tasks = [
                    generate_model_response(post_content, poster_name, simulator_persona, mt)
                    for mt in ["base", "grpo", "humanlm"]
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return {"base": results[0] if not isinstance(results[0], Exception) else "(Error)",
                        "grpo": results[1] if not isinstance(results[1], Exception) else "(Error)",
                        "humanlm": results[2] if not isinstance(results[2], Exception) else "(Error)"}

            def run_async_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(generate_all_responses())
                finally:
                    loop.close()

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_async_in_thread)
                responses = future.result(timeout=120)

        # Store generated persona in state from cache
        if cached_persona_data:
            state["generated_persona"] = cached_persona_data.get("generated_persona", {})
            state["persona_prompt"] = cached_persona_data.get("persona_prompt", "")
            state["persona_raw_output"] = cached_persona_data.get("persona_raw_output", "")
            state["simulator_prompt"] = cached_persona_data.get("simulator_prompt", "")
            print("\n" + "="*80)
            print("[ANNOTATE -> COMPARE] Storing Generated Persona in State:")
            print("="*80)
            print(json.dumps(state["generated_persona"], indent=2))
            print("="*80)
            print("\n[ANNOTATE -> COMPARE] Simulator Prompt (stored in state):")
            print("="*80)
            sim_prompt = state.get("simulator_prompt", "")
            print(sim_prompt)
            print("="*80 + "\n")

        # Store responses and model order
        model_types = ["base", "grpo", "humanlm"]
        shuffled_indices = list(range(3))
        random.shuffle(shuffled_indices)
        state["model_order"] = [model_types[i] for i in shuffled_indices]
        state["responses"] = responses

        print("\n" + "#"*80)
        print("#" + " "*78 + "#")
        print("#" + "  MODEL ORDER ASSIGNMENT (SHUFFLED)".center(78) + "#")
        print("#" + " "*78 + "#")
        print("#" + f"  Response A = {state['model_order'][0].upper()}".ljust(78) + "#")
        print("#" + f"  Response B = {state['model_order'][1].upper()}".ljust(78) + "#")
        print("#" + f"  Response C = {state['model_order'][2].upper()}".ljust(78) + "#")
        print("#" + " "*78 + "#")
        print("#"*80 + "\n")

        response_a = responses[state["model_order"][0]]
        response_b = responses[state["model_order"][1]]
        response_c = responses[state["model_order"][2]]

        state["response_a"] = response_a
        state["response_b"] = response_b
        state["response_c"] = response_c

        user_response_html = state.get("user_response_html", "")

        escaped_response_a = html.escape(response_a).replace('\n', '<br>')
        response_a_html = f"""
        <div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px; border: 2px solid #2980b9; min-height: 150px;'>
            <p style='color: black; line-height: 1.6;'>{escaped_response_a}</p>
        </div>
        """
        state["response_html_a"] = response_a_html

        escaped_response_b = html.escape(response_b).replace('\n', '<br>')
        state["response_html_b"] = f"""
        <div style='background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #27ae60; min-height: 150px;'>
            <p style='color: black; line-height: 1.6;'>{escaped_response_b}</p>
        </div>
        """

        escaped_response_c = html.escape(response_c).replace('\n', '<br>')
        state["response_html_c"] = f"""
        <div style='background-color: #f3e5f5; padding: 15px; border-radius: 8px; border: 2px solid #8e44ad; min-height: 150px;'>
            <p style='color: black; line-height: 1.6;'>{escaped_response_c}</p>
        </div>
        """

        # Build annotations HTML from self_description
        sd = state.get("self_description", {})
        annotations_html = f"""
        <div style='background-color: #f9f9f9; padding: 12px; border-radius: 6px; border: 1px solid #ddd;'>
            <table style='width: 100%; border-collapse: collapse; font-size: 13px;'>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding: 6px 8px; font-weight: 600; color: #2e7d32; width: 140px; vertical-align: top;'>Stance</td>
                    <td style='padding: 6px 8px; color: black;'>{html.escape(sd.get("stance", ""))}</td>
                </tr>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding: 6px 8px; font-weight: 600; color: #2e7d32; vertical-align: top;'>Emotion</td>
                    <td style='padding: 6px 8px; color: black;'>{html.escape(sd.get("emotion", ""))}</td>
                </tr>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding: 6px 8px; font-weight: 600; color: #2e7d32; vertical-align: top;'>Belief</td>
                    <td style='padding: 6px 8px; color: black;'>{html.escape(sd.get("belief", ""))}</td>
                </tr>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding: 6px 8px; font-weight: 600; color: #2e7d32; vertical-align: top;'>Value</td>
                    <td style='padding: 6px 8px; color: black;'>{html.escape(sd.get("value", ""))}</td>
                </tr>
                <tr style='border-bottom: 1px solid #eee;'>
                    <td style='padding: 6px 8px; font-weight: 600; color: #2e7d32; vertical-align: top;'>Goal</td>
                    <td style='padding: 6px 8px; color: black;'>{html.escape(sd.get("goal", ""))}</td>
                </tr>
                <tr>
                    <td style='padding: 6px 8px; font-weight: 600; color: #2e7d32; vertical-align: top;'>Communication Style</td>
                    <td style='padding: 6px 8px; color: black;'>{html.escape(sd.get("communication_style", ""))}</td>
                </tr>
            </table>
        </div>
        """

        # Debug label
        debug_html = ""
        if DEBUG_MODE:
            debug_html = f"""
            <div style='text-align: center; color: orange; font-weight: bold; margin-bottom: 10px;'>
                [DEBUG] A: {state["model_order"][0]} | B: {state["model_order"][1]} | C: {state["model_order"][2]}
            </div>
            """

        return (state, gr.update(visible=False), gr.update(visible=True),
                gr.update(value=state.get("post_html", "")),
                gr.update(value=user_response_html),  # overview user response
                gr.update(value=annotations_html),  # overview annotations
                gr.update(value=state["response_html_a"]),  # overview response A
                gr.update(value=state["response_html_b"]),  # overview response B
                gr.update(value=state["response_html_c"]),  # overview response C
                gr.update(value=user_response_html), gr.update(value=user_response_html), gr.update(value=user_response_html),
                gr.update(value=state["response_html_a"]),
                gr.update(value=state["response_html_b"]),
                gr.update(value=state["response_html_c"]),
                gr.update(value=debug_html))

    move_to_step2_button.click(
        fn=move_to_step2_button_click,
        inputs=[state, age_group, gender, occupation, location, nationality,
                rank_freedom, rank_health, rank_wealth, rank_success, rank_happiness,
                question1_answer, question2_answer, question3_answer, question4_answer,
                question5_answer, question6_answer, question7_answer, question8_answer],
        outputs=[state, pre_writing_window, step2_window, post_content_html]
    )

    # Step 2 Part 1 → Annotate Your Response
    continue_to_reflection_button.click(
        fn=continue_to_annotate_click,
        inputs=[state, user_response_textbox],
        outputs=[state, step2_window, step2_annotate_window,
                 annotate_post_display,
                 annotate_user_response_display]
    )

    # Annotate → Compare All Responses
    continue_to_compare_button.click(
        fn=continue_to_compare_click,
        inputs=[state, self_stance, self_emotion, self_belief, self_value, self_goal, self_communication],
        outputs=[state, step2_annotate_window, step2_part2_window,
                 step2_post_display_merged,
                 step2_overview_user_response,
                 step2_overview_annotations,
                 step2_overview_response_a, step2_overview_response_b, step2_overview_response_c,
                 step2_user_response_display_a, step2_user_response_display_b, step2_user_response_display_c,
                 step2_response_display_a, step2_response_display_b, step2_response_display_c,
                 step2_model_debug_labels]
    )

    # ============ STEP 2 NAVIGATION HANDLERS ============

    # Step 2 Part 2 (Merged) → Review
    def continue_to_review(state, comparison_a, comparison_b, comparison_c,
                           similarity_a, similarity_b, similarity_c,
                           ranking_reason, ranking_a, ranking_b, ranking_c,
                           humanlike_a, humanlike_b, humanlike_c, humanlikeness_reason):
        if not DEBUG_MODE and not state.get("cheat", ""):
            # Validate all three comparisons
            for name, comparison in [("A", comparison_a), ("B", comparison_b), ("C", comparison_c)]:
                word_count = len(comparison.split()) if comparison and comparison.strip() else 0
                if word_count < 50:
                    raise gr.Error(f"Please write at least 50 words for comparison with Response {name}. Current: {word_count} words.", duration=5)

            # Validate all similarity scores
            for name, score in [("A", similarity_a), ("B", similarity_b), ("C", similarity_c)]:
                if score is None or score < 1 or score > 10:
                    raise gr.Error(f"Please enter a valid similarity score (1-10) for Response {name}.", duration=5)

            # Validate ranking reason
            ranking_word_count = len(ranking_reason.split()) if ranking_reason and ranking_reason.strip() else 0
            if ranking_word_count < 15:
                raise gr.Error(f"Please write at least 15 words explaining your ranking. Current: {ranking_word_count} words.", duration=5)

            # Validate rankings
            ranks = [ranking_a, ranking_b, ranking_c]
            if None in ranks or "" in ranks:
                raise gr.Error("Please select a rank for all three responses.", duration=5)

            rank_values = [r.split()[0] if r else "" for r in ranks]  # Extract "1st", "2nd", "3rd"
            if len(set(rank_values)) != 3:
                raise gr.Error("Each rank (1st, 2nd, 3rd) must be used exactly once.", duration=5)

            # Validate all humanlikeness scores
            for name, score in [("A", humanlike_a), ("B", humanlike_b), ("C", humanlike_c)]:
                if score is None or score < 1 or score > 10:
                    raise gr.Error(f"Please enter a valid humanlikeness score (1-10) for Response {name}.", duration=5)

            # Validate humanlikeness reason
            humanlikeness_word_count = len(humanlikeness_reason.split()) if humanlikeness_reason and humanlikeness_reason.strip() else 0
            if humanlikeness_word_count < 10:
                raise gr.Error(f"Please write at least 10 words explaining your human-likeness scores. Current: {humanlikeness_word_count} words.", duration=5)

        # Store all comparison data
        state["comparison_a"] = comparison_a
        state["comparison_b"] = comparison_b
        state["comparison_c"] = comparison_c
        state["overall_similarity_a"] = similarity_a
        state["overall_similarity_b"] = similarity_b
        state["overall_similarity_c"] = similarity_c
        state["ranking_reason"] = ranking_reason
        state["similarity_rankings"] = {"a": ranking_a, "b": ranking_b, "c": ranking_c}
        state["humanlikeness_a"] = humanlike_a
        state["humanlikeness_b"] = humanlike_b
        state["humanlikeness_c"] = humanlike_c
        state["humanlikeness_reason"] = humanlikeness_reason

        print("\n" + "="*80)
        print("[COMPARE -> REVIEW] User Comparison & Scores:")
        print("="*80)
        for name, comp, sim, hl in [("A", comparison_a, similarity_a, humanlike_a),
                                     ("B", comparison_b, similarity_b, humanlike_b),
                                     ("C", comparison_c, similarity_c, humanlike_c)]:
            print(f"\nResponse {name} (similarity={sim}, humanlikeness={hl}):")
            print(f"  Comparison: {comp}")
        print(f"\nRanking: A={ranking_a}, B={ranking_b}, C={ranking_c}")
        print(f"Ranking reason: {ranking_reason}")
        print(f"Humanlikeness reason: {humanlikeness_reason}")
        print("="*80 + "\n")

        # Format comparison texts for review
        def format_comparison(text):
            escaped = html.escape(text).replace('\n', '<br>') if text else "Not provided"
            return f"<div style='background-color: #f5f5f5; padding: 10px; border-radius: 8px;'><p style='color: black;'>{escaped}</p></div>"

        return (state, gr.update(visible=False), gr.update(visible=True),
                gr.update(value=state.get("post_html", "")),
                gr.update(value=state.get("user_response_html", "")),
                gr.update(value=state.get("response_html_a", "")),
                gr.update(value=state.get("response_html_b", "")),
                gr.update(value=state.get("response_html_c", "")),
                gr.update(value=format_comparison(comparison_a)),
                gr.update(value=format_comparison(comparison_b)),
                gr.update(value=format_comparison(comparison_c)),
                gr.update(value=humanlike_a),
                gr.update(value=humanlike_b),
                gr.update(value=humanlike_c),
                gr.update(value=similarity_a),
                gr.update(value=similarity_b),
                gr.update(value=similarity_c))

    continue_to_review_button.click(
        fn=continue_to_review,
        inputs=[state, comparison_text_a, comparison_text_b, comparison_text_c,
                overall_similarity_a, overall_similarity_b, overall_similarity_c,
                ranking_reason_textbox, rank_a, rank_b, rank_c,
                humanlikeness_score_a, humanlikeness_score_b, humanlikeness_score_c,
                humanlikeness_reason_textbox],
        outputs=[state, step2_part2_window, step2_review_window,
                 step2_review_post_display,
                 step2_review_user_response,
                 step2_review_response_a, step2_review_response_b, step2_review_response_c,
                 step2_review_comparison_a, step2_review_comparison_b, step2_review_comparison_c,
                 review_humanlikeness_a, review_humanlikeness_b, review_humanlikeness_c,
                 review_similarity_a, review_similarity_b, review_similarity_c]
    )

    # Step 2 Review → Final Submission
    # Step 2 Review → Final Submission
    def proceed_to_submit(state, sim_a, sim_b, sim_c, human_a, human_b, human_c, additional_feedback):
        if not DEBUG_MODE and not state.get("cheat", ""):
            # Validate scores
            for name, score in [("Similarity A", sim_a), ("Similarity B", sim_b), ("Similarity C", sim_c),
                                ("Humanlikeness A", human_a), ("Humanlikeness B", human_b), ("Humanlikeness C", human_c)]:
                if score is None or score < 1 or score > 10:
                    raise gr.Error(f"Please enter a valid score (1-10) for {name}.", duration=5)

        # Store final Step 2 scores
        state["final_similarity_a"] = sim_a
        state["final_similarity_b"] = sim_b
        state["final_similarity_c"] = sim_c
        state["final_humanlikeness_a"] = human_a
        state["final_humanlikeness_b"] = human_b
        state["final_humanlikeness_c"] = human_c

        # Get rankings from state (already stored in continue_to_review)
        rankings = state.get("similarity_rankings", {})
        state["similarity_rank_a"] = rankings.get("a", "")
        state["similarity_rank_b"] = rankings.get("b", "")
        state["similarity_rank_c"] = rankings.get("c", "")

        # Build simplified evaluation dicts
        state["evaluation_a"] = {
            "similarity_rating": {"overall": sim_a},
            "humanlikeness_rating": human_a,
            "comparison_text": state.get("comparison_a", "")
        }
        state["evaluation_b"] = {
            "similarity_rating": {"overall": sim_b},
            "humanlikeness_rating": human_b,
            "comparison_text": state.get("comparison_b", "")
        }
        state["evaluation_c"] = {
            "similarity_rating": {"overall": sim_c},
            "humanlikeness_rating": human_c,
            "comparison_text": state.get("comparison_c", "")
        }

        state["additional_feedback"] = additional_feedback
        state["end_time"] = time.time()
        state["time_spend"] = state["end_time"] - state["start_time"]

        return (state, gr.update(visible=False), gr.update(visible=True))

    proceed_to_step3_button.click(
        fn=proceed_to_submit,
        inputs=[state, review_similarity_a, review_similarity_b, review_similarity_c,
                review_humanlikeness_a, review_humanlikeness_b, review_humanlikeness_c,
                additional_feedback_textbox],
        outputs=[state, step2_review_window, final_button_col]
    )

    # ============ STEP 2 BACK BUTTON HANDLERS ============

    def go_back_to_part2(state):
        """Go back from Review to Part 2 (Compare page), restoring previous values."""
        rankings = state.get("similarity_rankings", {})
        return (gr.update(visible=False), gr.update(visible=True),
                gr.update(value=state.get("comparison_a", "")),
                gr.update(value=state.get("comparison_b", "")),
                gr.update(value=state.get("comparison_c", "")),
                gr.update(value=state.get("overall_similarity_a")),
                gr.update(value=state.get("overall_similarity_b")),
                gr.update(value=state.get("overall_similarity_c")),
                gr.update(value=state.get("ranking_reason", "")),
                gr.update(value=rankings.get("a")),
                gr.update(value=rankings.get("b")),
                gr.update(value=rankings.get("c")),
                gr.update(value=state.get("humanlikeness_a")),
                gr.update(value=state.get("humanlikeness_b")),
                gr.update(value=state.get("humanlikeness_c")),
                gr.update(value=state.get("humanlikeness_reason", "")))

    back_to_part2_button.click(
        fn=go_back_to_part2,
        inputs=[state],
        outputs=[step2_review_window, step2_part2_window,
                 comparison_text_a, comparison_text_b, comparison_text_c,
                 overall_similarity_a, overall_similarity_b, overall_similarity_c,
                 ranking_reason_textbox, rank_a, rank_b, rank_c,
                 humanlikeness_score_a, humanlikeness_score_b, humanlikeness_score_c,
                 humanlikeness_reason_textbox]
    )


    post_hit_js = """
        function(state) {
            // If there is an assignmentId, then the submitter is on mturk
            // and has accepted the HIT. So, we need to submit their HIT.
            
            const form = document.createElement('form');
            const turkSubmitTo = state.turkSubmitTo || "https://www.mturk.com";
            form.action = `${turkSubmitTo}/mturk/externalSubmit`;
            form.method = 'post';
            for (const key in state) {
                const hiddenField = document.createElement('input');
                hiddenField.type = 'hidden';
                hiddenField.name = key;
                hiddenField.value = state[key];
                form.appendChild(hiddenField);
            };
            document.body.appendChild(form);
            form.submit();
            return state;
        }
        """
    
    refresh_webpage_js = """
        function(state) {
            // Parse the URL parameters
            const urlParams = new URLSearchParams(window.location.search);
            
            // Construct the new URL
            const newUrl = window.location.origin + window.location.pathname + '?' + urlParams.toString();
            
            // Redirect to the new URL
            window.location.href = newUrl;
            
            return state;
        }
    """

    remove_problem_bank_js = """
        function(state) {
            var problemBank = document.getElementById("problem-bank");
            if (problemBank) {
                problemBank.remove();
            }
            return state;
        }
        """
    
    ##  submit to huggingface - simplified
    def submit_simple(state):
        # Wait for background persona generation to complete before saving
        thread_key = state.get("persona_thread_key")
        if thread_key and thread_key in persona_generation_threads:
            persona_thread = persona_generation_threads[thread_key]
            if persona_thread.is_alive():
                print("Waiting for background persona generation to complete...")
                persona_thread.join(timeout=30)  # Wait up to 30 seconds
                if persona_thread.is_alive():
                    print("Warning: Persona generation timed out, proceeding without it")
            # Clean up thread reference
            del persona_generation_threads[thread_key]

        # Ensure we have the generated persona from the global cache
        if thread_key and thread_key in generated_personas_cache:
            cached_persona_data = generated_personas_cache[thread_key]
            if not state.get("generated_persona"):
                state["generated_persona"] = cached_persona_data.get("generated_persona", {})
                state["persona_prompt"] = cached_persona_data.get("persona_prompt", "")
                state["persona_raw_output"] = cached_persona_data.get("persona_raw_output", "")
            # Clean up cache
            del generated_personas_cache[thread_key]

        # All data is already in state from the evaluation step
        state["submission_time"] = time.time()

        # Save the state to file and upload to HuggingFace
        if not state.get("assignmentId"):
            gr.Info("Thank you for your participation! Your response has been submitted.", duration=2)

            filename = f"{state.get('user_id', 'unknown')}_{state.get('post_id', 'unknown')}"
            filename = sanitize_filename(filename)

            file_path = os.path.join(FOLDER_PATH, state.get("username", "anonymous"), f"{filename}.json")
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4)
            print(f"[SUBMIT] Saved local JSON: {file_path}")

            try:
                hf_api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{state.get('username', 'anonymous')}/{filename}.json",
                    repo_id=DATASET_REPO_URL,
                    repo_type="dataset",
                )
                print(f"[SUBMIT] Uploaded JSON to HuggingFace: {DATASET_REPO_URL}/{state.get('username', 'anonymous')}/{filename}.json")
            except Exception as e:
                print(f"[SUBMIT] ERROR uploading JSON to HuggingFace: {e}")
                logger.error(f"Error uploading JSON to HuggingFace: {e}")
        else:
            gr.Info("Thank you for your participation! Your response is being submitted.", duration=2)

            filename = f"{state['assignmentId']}_{state.get('post_id', 'unknown')}"
            filename = sanitize_filename(filename)

            file_path = os.path.join(FOLDER_PATH, state.get("username", "anonymous"), f"{filename}.json")
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4)
            print(f"[SUBMIT] Saved local JSON (mturk): {file_path}")

            try:
                hf_api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"{state.get('username', 'anonymous')}/{filename}.json",
                    repo_id=DATASET_REPO_URL,
                    repo_type="dataset",
                )
                print(f"[SUBMIT] Uploaded JSON to HuggingFace: {DATASET_REPO_URL}/{state.get('username', 'anonymous')}/{filename}.json")
            except Exception as e:
                print(f"[SUBMIT] ERROR uploading JSON to HuggingFace: {e}")
                logger.error(f"Error uploading JSON to HuggingFace: {e}")

        if state.get("username") == "mturk":
            logger.info(f"<><>submit AssignmentId: {state.get('assignmentId')}, WorkerId: {state.get('workerId')}, Document Type: {state.get('post_id')}, has been submitted.")
        else:
            logger.info(f"<><>submit Username: {state.get('username')}, Document Type: {state.get('post_id')}, has been submitted.")

        # Local JSON files are already saved above to FOLDER_PATH with all state data
        # including: user_profile, persona_prompt, persona_raw_output, responses, model_order, evaluations

        # Append to HuggingFace CSV file
        try:
            worker_id = state.get("workerId") or state.get("username") or state.get("user_id", "unknown")
            append_to_hf_csv(
                hf_api_instance=hf_api,
                repo_id=DATASET_REPO_URL,
                worker_id=worker_id,
                user_profile=state.get("user_profile"),
                persona_prompt=state.get("persona_prompt"),
                persona_output=state.get("persona_raw_output"),
                user_response=state.get("user_post_response"),
                model_responses=state.get("responses"),
                model_order=state.get("model_order"),
                evaluations={
                    "evaluation_a": state.get("evaluation_a"),
                    "evaluation_b": state.get("evaluation_b"),
                    "evaluation_c": state.get("evaluation_c"),
                    "humanlikeness_reason": state.get("humanlikeness_reason"),
                    "overall_preference": state.get("overall_preference"),
                    "additional_feedback": state.get("additional_feedback")
                }
            )
            logger.info(f"Appended to HuggingFace CSV: {DATASET_REPO_URL}/summarize.csv")
        except Exception as e:
            logger.error(f"Error appending to HuggingFace CSV: {e}")

        return state

    submit_hf_button.click(
        submit_simple,
        inputs=[state],
        outputs=[state]
    ).success(
        lambda state: state, inputs=[state], outputs=[state], js=refresh_webpage_js)

    ## submit to mturk
    submit_hit_button.click(
        submit_simple,
        inputs=[state],
        outputs=[state]
    ).success(lambda state: state, inputs=[state], outputs=[state], js=post_hit_js)
    
    
    cookie_js = '''
        function(value){
            let user_id = value['user_id']; // Access the user_id from the value dictionary
            document.cookie = 'user_id=' + user_id + '; Path=/;  SameSite=None; Secure'; // this allows iframe like in amt
            return value;
        }
    '''


    demo.load(load_instance, None, outputs=[state, user_id,
                                            submit_hf_button, 
                                            submit_hit_button]).then(
                            lambda user_id: None, inputs=[user_id], js=cookie_js)
    
    
if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4, max_size=4)
    demo.launch(max_threads=100, share=True)
