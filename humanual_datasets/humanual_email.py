"""
Enron Email Dataset Processor
Processes Enron email CSV into the expected format for process_raw.py

Original dataset: Enron email corpus
"""

import argparse
import csv
import email.utils
import json
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional
from datasets import Dataset, Features, Value
from dotenv import load_dotenv
from tqdm import tqdm

from .utils_parser import memory_usage, convert_to_nested_json, parse_date

# Increase CSV field size limit to handle large email messages
csv.field_size_limit(10000000)  # 10MB limit

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


def parse_email_from_nested(nested_email: Dict, file_path: str) -> Optional[Dict]:
    """
    Convert nested email structure to the format expected by the dataset.
    Extracts metadata and converts to email dict format.
    """
    if not nested_email:
        return None
    
    headers = nested_email.get('headers', {})
    body = nested_email.get('body', '').strip()
    
    # Extract fields from headers
    message_id = headers.get('Message-ID', '').strip()
    from_addr = headers.get('From', '')
    to_addr = headers.get('To', '')
    subject = headers.get('Subject', '')
    date_str = headers.get('Date', '') or headers.get('Sent', '')
    
    # Parse timestamp
    timestamp = None
    if date_str:
        timestamp_float = parse_date(date_str)
        if timestamp_float:
            timestamp = int(timestamp_float)
    
    # Also try email.utils parsing as fallback
    if not timestamp and date_str:
        try:
            date_tuple = email.utils.parsedate_tz(date_str)
            if date_tuple:
                timestamp = int(email.utils.mktime_tz(date_tuple))
        except Exception:
            pass
    
    # Extract user_id from From field (already cleaned to just email)
    user_id = from_addr.lower() if from_addr else None
    
    # Build metadata dict
    metadata = {
        "message_id": message_id,
        "from": from_addr,
        "to": to_addr,
        "subject": subject,
        "in_reply_to": headers.get('In-Reply-To', '').strip(),
        "references": headers.get('References', '').strip(),
        "cc": headers.get('Cc', ''),
        "bcc": headers.get('Bcc', ''),
        "x_from": headers.get('X-From', ''),
        "x_to": headers.get('X-To', ''),
        "x_folder": headers.get('X-Folder', ''),
        "x_origin": headers.get('X-Origin', ''),
        "x_filename": headers.get('X-FileName', ''),
        "body_length": len(body),
        "file": file_path,
    }
    
    return {
        "message_id": message_id,
        "timestamp": timestamp,
        "user_id": user_id,
        "subject": subject,
        "body": body,
        "from_addr": from_addr,
        "to_addr": to_addr,
        "in_reply_to": headers.get('In-Reply-To', '').strip(),
        "references": headers.get('References', '').strip(),
        "metadata": metadata,
    }


def flatten_nested_email_thread(nested_json: Dict, file_path: str) -> List[Dict]:
    """
    Flatten nested email structure into chronological list of emails.
    Returns list of email dicts in chronological order (oldest first).
    """
    if not nested_json:
        return []
    
    emails = []
    
    # Add outer email
    outer_email = parse_email_from_nested(nested_json.get('outer_email', {}), file_path)
    if outer_email:
        emails.append(outer_email)
    
    # Recursively flatten nested emails
    def flatten_nested(nested_list):
        for nested in nested_list:
            email_dict = parse_email_from_nested(nested, file_path)
            if email_dict:
                emails.append(email_dict)
            # Recursively process nested emails
            if 'nested_emails' in nested:
                flatten_nested(nested.get('nested_emails', []))
    
    flatten_nested(nested_json.get('nested_emails', []))
    
    return emails


def normalize_subject(subject: str) -> str:
    """Normalize email subject for threading (remove Re:, Fwd:, etc.)."""
    if not subject:
        return ""
    # Remove common reply/forward prefixes
    normalized = subject.replace("Re:", "").replace("RE:", "").replace("re:", "")
    normalized = normalized.replace("Fwd:", "").replace("FWD:", "").replace("fwd:", "")
    normalized = normalized.replace("FW:", "").replace("fw:", "")
    normalized = normalized.replace("Fw:", "").replace("fW:", "")
    # Remove extra whitespace
    normalized = " ".join(normalized.split())
    return normalized.strip()


def _content_dedupe_key(email_data: Dict) -> str:
    """
    Many Enron exports contain the same email multiple times in different folders.
    Message-ID is usually unique; but duplicates can appear as distinct Message-IDs with identical content.
    We dedupe using (from + timestamp + body_hash) as a secondary key.
    """
    frm = (email_data.get("user_id") or "").lower()
    ts = int(email_data.get("timestamp") or 0)
    body = email_data.get("body") or ""
    bh = hashlib.md5(body.encode("utf-8", errors="ignore")).hexdigest()
    return f"{frm}|{ts}|{bh}"


def _make_post_id(root_email: Dict, normalized_subject: str) -> str:
    """
    post_id should be unique per conversation thread (but shared by all reply-entries in that thread).
    Use a stable hash derived from the thread root + subject.
    """
    root_mid = (root_email.get("message_id") or "").strip()
    root_ts = int(root_email.get("timestamp") or 0)
    seed = f"{normalized_subject}|{root_mid}|{root_ts}".encode("utf-8", errors="ignore")
    h = hashlib.md5(seed).hexdigest()[:16]
    return f"enron_{h}"


def _split_by_time_gap(emails: List[Dict], gap_seconds: int) -> List[List[Dict]]:
    if not emails:
        return []
    emails = sorted(emails, key=lambda e: e.get("timestamp") or 0)
    threads: List[List[Dict]] = []
    cur: List[Dict] = [emails[0]]
    for e in emails[1:]:
        prev_ts = int(cur[-1].get("timestamp") or 0)
        ts = int(e.get("timestamp") or 0)
        if ts - prev_ts > gap_seconds:
            threads.append(cur)
            cur = [e]
        else:
            cur.append(e)
    threads.append(cur)
    return threads


class EnronEmailDataset:
    def __init__(
        self,
        csv_path: str,
        push_to_hub: Optional[str] = None,
        config_path: Optional[str] = None,
        min_thread_size: int = 2,  # Default to 2: need at least post + one reply
        max_rows: Optional[int] = None,  # For local spot-checks; None means full file
    ):
        self.csv_path = csv_path
        self.push_to_hub = push_to_hub
        self.min_thread_size = min_thread_size
        self.max_rows = max_rows
        
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()

    def create_raw_dataset(self):
        """Process Enron emails from CSV into the expected format."""
        print(f"[{datetime.now()}] Loading emails from {self.csv_path}...")
        
        # First pass: parse all emails
        emails: List[Dict] = []
        msgid_seen = set()
        content_seen = set()
        
        with open(self.csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(tqdm(reader, desc="Parsing emails")):
                if self.max_rows is not None and i >= self.max_rows:
                    break
                
                file_path = row.get("file", "")
                message_str = row["message"]
                
                # Use nested parsing to extract email threads
                nested_json = convert_to_nested_json(file_path, message_str)
                
                if not nested_json:
                    continue
                
                # Flatten nested structure into chronological list
                thread_emails = flatten_nested_email_thread(nested_json, file_path)
                
                if not thread_emails:
                    continue
                
                # Process each email in the thread
                for email_data in thread_emails:
                    # Skip if no body or too short
                    if not email_data["body"] or len(email_data["body"].strip()) < 10:
                        continue
                    
                    # Skip if no user_id
                    if not email_data["user_id"]:
                        continue
                    
                    # Skip if no timestamp
                    if not email_data["timestamp"]:
                        continue

                    # De-dupe:
                    # - Primary: Message-ID
                    # - Secondary: (from + timestamp + body_hash) for content duplicates across folders
                    message_id = (email_data.get("message_id") or "").strip()
                    if message_id:
                        if message_id in msgid_seen:
                            continue
                        msgid_seen.add(message_id)

                    ckey = _content_dedupe_key(email_data)
                    if ckey in content_seen:
                        continue
                    content_seen.add(ckey)

                    emails.append(email_data)
        
        print(
            f"[{datetime.now()}] Parsed {len(emails)} valid emails "
            f"(deduped by Message-ID and content-key)"
        )
        
        # Second pass: build thread relationships from nested structures.
        # Group emails by file_path (each nested structure from utils_parser is already a thread)
        # and also use subject + time-gap splitting as fallback
        print(f"[{datetime.now()}] Building email threads from nested structures...")

        # Group emails by file_path (each nested structure is already a thread)
        by_file: Dict[str, List[Dict]] = defaultdict(list)
        for e in emails:
            file_path = e.get("metadata", {}).get("file", "") if isinstance(e.get("metadata"), dict) else ""
            if not file_path:
                # Fallback: use subject grouping
                norm_subj = normalize_subject(e.get("subject") or "")
                if not norm_subj:
                    norm_subj = "(no subject)"
                by_file[norm_subj].append(e)
            else:
                by_file[file_path].append(e)

        # Split buckets into unique threads and assign unique post_id per thread
        gap_seconds = 7 * 24 * 3600  # 7 days default heuristic
        threads: Dict[str, List[Dict]] = {}
        for file_or_subj, file_emails in by_file.items():
            # Sort by timestamp
            file_emails.sort(key=lambda e: e.get("timestamp") or 0)
            
            # If multiple emails from same file, split by time gaps
            subthreads = _split_by_time_gap(file_emails, gap_seconds=gap_seconds)
            for t in subthreads:
                if not t:
                    continue
                norm_subj = normalize_subject(t[0].get("subject") or "")
                if not norm_subj:
                    norm_subj = "(no subject)"
                post_id = _make_post_id(t[0], norm_subj)
                # Ensure uniqueness even in pathological collisions
                if post_id in threads:
                    post_id = _make_post_id(t[0], norm_subj + "|alt")
                threads[post_id] = t

        # Filter threads by minimum size
        valid_threads = {pid: t for pid, t in threads.items() if len(t) >= self.min_thread_size}
        print(f"[{datetime.now()}] Found {len(valid_threads)} threads with >= {self.min_thread_size} emails")
        
        # Third pass: create dataset entries
        print(f"[{datetime.now()}] Creating dataset entries...")
        all_entries = []
        
        for post_id, thread_emails in tqdm(valid_threads.items(), desc="Processing threads"):
            # Thread is already sorted by timestamp (temporal order for turn-based conversation)
            # Skip the first email (original post) - we only create entries for replies
            # For each reply email in the thread, create an entry
            for idx, current_email in enumerate(thread_emails):
                if idx == 0:
                    # Skip the first email - it's the post, not a reply
                    continue
                
                # Build prompt: original post + all previous replies
                prompt = []
                
                # The "post" is always the first email in the thread
                first_email = thread_emails[0]
                post_prompt = {
                    "role": first_email["user_id"],
                    "content": first_email["body"],  # Just the body, subject is in metadata
                    "metadata": json.dumps(first_email["metadata"]),
                }
                prompt.append(post_prompt)
                
                # Add all previous replies in temporal order (turn-based: x replied to y)
                for prev_email in thread_emails[1:idx]:
                    # Each reply shows who replied (role = sender) and what they said
                    reply_prompt = {
                        "role": prev_email["user_id"],
                        "content": prev_email["body"],  # Just the body, subject already in post
                        "metadata": json.dumps(prev_email["metadata"]),
                    }
                    prompt.append(reply_prompt)
                
                # Completion is the current email's body (the reply we're predicting)
                completion = current_email["body"]

                # Guardrails:
                # - completion should not be identical to the last prompt message
                # - completion should be non-trivial after cleaning
                if not completion or len(completion.strip()) < 10:
                    continue
                if prompt and completion.strip() == (prompt[-1]["content"] or "").strip():
                    continue
                
                entry = {
                    "prompt": prompt,
                    "completion": completion,
                    "post_id": post_id,  # Unique per conversation thread
                    "user_id": current_email["user_id"],
                    "timestamp": current_email["timestamp"],
                    "metadata": json.dumps(current_email["metadata"]),
                }
                all_entries.append(entry)
        
        print(f"[{datetime.now()}] Created {len(all_entries)} dataset entries")
        print(f"[{datetime.now()}] Memory before converting to Dataset: {memory_usage()}")
        
        # Convert to Dataset
        dataset = Dataset.from_list(all_entries, features=FEATURES)
        print(f"[{datetime.now()}] Memory after creating Dataset: {memory_usage()}")
        
        if self.push_to_hub:
            print(f"[{datetime.now()}] Pushing dataset to hub: {self.push_to_hub}")
            # push_to_hub will overwrite existing data by default if same config/split
            dataset.push_to_hub(
                self.push_to_hub, 
                config_name="default", 
                split="split0", 
                private=True,
                commit_message="Update Enron email dataset"
            )
            print(f"[{datetime.now()}] Pushed dataset to hub")
        
        return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enron Email Dataset Processor")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the Enron emails CSV file",
    )
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
        "--min_thread_size",
        type=int,
        default=2,
        help="Minimum number of emails per thread to include (default: 2, need at least post + one reply)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="For local spot-checks: only read the first N CSV rows (default: read full file).",
    )
    parser.add_argument(
        "--save_to_disk",
        type=str,
        default=None,
        help="If set, save the resulting HF Dataset locally via Dataset.save_to_disk(path).",
    )
    args = parser.parse_args()

    ds_builder = EnronEmailDataset(
        csv_path=args.csv_path,
        push_to_hub=args.push_to_hub,
        config_path=args.config,
        min_thread_size=args.min_thread_size,
        max_rows=args.max_rows,
    )

    ds = ds_builder.create_raw_dataset()
    if args.save_to_disk:
        print(f"[{datetime.now()}] Saving dataset to disk: {args.save_to_disk}")
        ds.save_to_disk(args.save_to_disk)
        print(f"[{datetime.now()}] Saved dataset to disk")

