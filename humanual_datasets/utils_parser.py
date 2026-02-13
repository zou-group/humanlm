#!/usr/bin/env python3
"""
Shared utilities and email parsing functions.

Part 1: General utility functions (message parsing, progress bars, memory tracking).
Part 2: Enron email parser (CSV to nested JSON structure).
"""

import argparse
import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import datasets
import datasets.data_files
import datasets.download
import datasets.io.csv
import datasets.io.json
import datasets.io.parquet
import datasets.io.sql
import datasets.search
import psutil
from tqdm import tqdm
from tqdm.rich import tqdm as rich_tqdm
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    filesize,
)
from rich.text import Text


# =============================================================================
# Part 1: General Utilities
# =============================================================================

RST = "\033[0m"
BRED = "\033[1;31m"
BGREEN = "\033[1;32m"
BYELLOW = "\033[1;33m"
BRI = "\033[1m"


def parse_messages(messages, strip_sys_prompt=True):
    """
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'},
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    """
    if messages is None:
        return ""

    def strip_system_prompt(messages):
        return [msg for msg in messages if msg["role"] != "system"]

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    chat = "\n".join(f"Message from {m['role']}: \n'''\n{m['content']}\n'''" for m in messages)

    return chat


def parse_messages_title(messages, strip_sys_prompt=True):
    """
    For Medium-style threads:
      - For the first non-system message: use (title, subtitle) from metadata.
      - For remaining messages: use the regular message content (like parse_messages).
    All joined into a single context string.
    """
    if messages is None:
        return ""

    def strip_system_prompt(msgs):
        return [m for m in msgs if m.get("role") != "system"]

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    if not messages:
        return ""

    def get_meta_dict(m):
        meta = m.get("metadata")
        if isinstance(meta, dict):
            return meta
        if isinstance(meta, str):
            meta = meta.strip()
            if not meta:
                return {}
            try:
                return json.loads(meta)
            except json.JSONDecodeError:
                return {}
        return {}

    first = messages[0]
    meta = get_meta_dict(first)

    counts = meta.get("counts", {}) or {}
    title = counts.get("title") or meta.get("title") or ""
    subtitle = counts.get("subtitle") or meta.get("subtitle") or ""

    role = first.get("role", "unknown")
    lines = [
        f"Article from {role}: \n'''\nTitle: {title}\nSubtitle: {subtitle}\n'''"
    ]

    rest = messages[1:]
    if rest:
        # Reuse existing parse_messages, but don't strip system again
        rest_str = parse_messages(rest, strip_sys_prompt=False)
        if rest_str:
            lines.append(rest_str)

    return "\n".join(lines)


def inplace_clean_empty_dict(meta: Any, exclude_fields: set[str] = None) -> Any:
    """Clean other_meta_data by removing unwanted fields and empty dicts"""
    if type(meta) is list:
        for v in meta:
            inplace_clean_empty_dict(v, exclude_fields)
    elif type(meta) is dict:
        if exclude_fields:
            for k in exclude_fields:
                meta.pop(k, None)
        empty_v_key_list = [k for k, v in meta.items() if type(v) is dict and len(v) == 0]
        for k in empty_v_key_list:
            meta.pop(k, None)
        for v in meta.values():
            inplace_clean_empty_dict(v, exclude_fields)


def size_str(size_in_bytes: int) -> str:
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < (1024**2):
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < (1024**3):
        return f"{size_in_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_in_bytes / (1024 ** 3):.2f} GB"


def memory_usage() -> dict:
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {"rss": size_str(mem_info.rss), "vms": size_str(mem_info.vms)}


class my_tqdm_for_hf(rich_tqdm):
    """
    Class to override `disable` argument in case progress bars are globally disabled.

    Taken from https://github.com/tqdm/tqdm/issues/619#issuecomment-619639324.
    """

    def __init__(self, *args, **kwargs):
        if datasets.utils.are_progress_bars_disabled():
            kwargs["disable"] = True
        # Disable rich progress to avoid conflicts
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

    def __delattr__(self, attr: str) -> None:
        """Fix for https://github.com/huggingface/datasets/issues/6066"""
        try:
            super().__delattr__(attr)
        except AttributeError:
            if attr != "_lock":
                raise


def patch_datasets_tqdm():
    datasets.arrow_dataset.hf_tqdm = my_tqdm_for_hf
    datasets.arrow_reader.hf_tqdm = my_tqdm_for_hf
    datasets.builder.hf_tqdm = my_tqdm_for_hf
    datasets.data_files.hf_tqdm = my_tqdm_for_hf
    datasets.download.download_manager.tqdm = my_tqdm_for_hf
    datasets.io.csv.hf_tqdm = my_tqdm_for_hf
    datasets.io.json.hf_tqdm = my_tqdm_for_hf
    datasets.io.parquet.hf_tqdm = my_tqdm_for_hf
    datasets.io.sql.hf_tqdm = my_tqdm_for_hf
    datasets.search.hf_tqdm = my_tqdm_for_hf

    try:
        import huggingface_hub.utils._xet_progress_reporting

        huggingface_hub.utils._xet_progress_reporting.tqdm = rich_tqdm
    except ModuleNotFoundError as e:
        # NOTE: Given `_xet_progress_reporting` is private, it may not exist in future versions.
        pass


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit="", unit_scale=False, unit_divisor=1000):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            return Text(f"? {self.unit}/s", style="progress.data.speed")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        precision = 0 if unit == 1 else 1
        return Text(f"{speed/unit:,.{precision}f} {suffix}{self.unit}/s", style="progress.data.speed")


def make_progress(description: str) -> Progress:
    return Progress(
        TextColumn(description),
        "[progress.percentage]{task.percentage:>4.0f}%",
        BarColumn(),
        MofNCompleteColumn(),
        "[",
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        ",",
        RateColumn(),
        "]",
    )


# =============================================================================
# Part 2: Enron Email Parser
# =============================================================================

# Increase CSV field size limit to handle large email messages
csv.field_size_limit(10000000)  # 10MB limit

# Pre-compile regex patterns for better performance
FORWARDED_PATTERN = re.compile(r'-{5,}.*Forwarded by.*-{5,}', re.IGNORECASE | re.DOTALL)
NESTED_FROM_PATTERN = re.compile(r'^".*"\s*<[^>]+>\s+on\s+', re.IGNORECASE)
EMAIL_HEADER_PATTERN = re.compile(r'^(From|To|Subject|Date|Message-ID|Sent):', re.IGNORECASE)
DASHES_PATTERN = re.compile(r'-{5,}')


def clean_email_field(value: str) -> str:
    """
    Clean an email field value using regex to extract just the email address.
    Handles formats like:
    - "email@domain.com"
    - "Name" <email@domain.com>
    - <email@domain.com>
    - email@domain.com [mailto:email@domain.com]
    - email@domain.com]
    """
    if not value:
        return value

    # First, try to extract email address directly using regex
    email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    email_match = re.search(email_pattern, value)

    if email_match:
        return email_match.group(1)

    # If no email found, return cleaned version (remove brackets, etc.)
    cleaned = value.rstrip(']').strip()
    # Remove angle brackets if present
    if cleaned.startswith('<') and cleaned.endswith('>'):
        cleaned = cleaned[1:-1].strip()
    # Try one more time after cleaning
    email_match = re.search(email_pattern, cleaned)
    if email_match:
        return email_match.group(1)

    return cleaned


def parse_email_headers(lines: List[str], start_idx: int) -> Tuple[Dict[str, str], int]:
    """
    Parse email headers starting from start_idx.
    Returns (headers_dict, body_start_index)
    """
    headers = {}
    i = start_idx

    # Standard email header fields (case-insensitive)
    valid_header_fields = {
        'message-id', 'date', 'from', 'to', 'subject', 'cc', 'bcc',
        'mime-version', 'content-type', 'content-transfer-encoding',
        'x-from', 'x-to', 'x-cc', 'x-bcc', 'x-folder', 'x-origin', 'x-filename',
        'sent', 'reply-to', 'references', 'in-reply-to', 'return-path',
        'received', 'sender', 'list-id', 'list-unsubscribe'
    }

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Empty line after headers indicates body start
        if stripped == '':
            if headers:  # Only stop if we've found some headers
                return headers, i + 1
            else:
                # Skip empty lines at the start
                i += 1
                continue

        # Check if this is a header line (key: value format)
        if ':' in line and not stripped.startswith('>'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                key_lower = key.lower()

                # Only accept valid email header fields
                # Also accept X-* headers (custom headers)
                if key_lower in valid_header_fields or key_lower.startswith('x-'):
                    value = parts[1].strip()
                    # Clean email fields using regex
                    if key_lower in ['from', 'to', 'cc']:
                        value = clean_email_field(value)
                    headers[key] = value
                else:
                    # If we've already parsed some headers and hit an invalid header,
                    # we've likely reached the body
                    if headers:
                        return headers, i
                    # Otherwise, this might be the start of the body, so stop parsing headers
                    break

        i += 1

    return headers, i


def parse_nested_from_line(line: str) -> Dict[str, str]:
    """
    Parse a nested email "From" line like: "Name" <email@domain.com> on date
    Returns dict with 'From' and 'Date' keys
    """
    result = {}

    # Extract email address using regex
    email_pattern = r'<([^>]+)>'
    email_match = re.search(email_pattern, line)

    # Extract email address - always return just the clean email, not the name format
    if '"' in line and '<' in line:
        name_email_match = re.search(r'"([^"]+)"\s*<([^>]+)>', line)
        if name_email_match:
            # Extract and clean just the email address
            result['From'] = clean_email_field(name_email_match.group(2))
        elif email_match:
            result['From'] = clean_email_field(email_match.group(1))
    elif email_match:
        result['From'] = clean_email_field(email_match.group(1))

    # Extract date (after "on")
    date_match = re.search(r'on\s+(.+)', line, re.IGNORECASE)
    if date_match:
        result['Date'] = date_match.group(1).strip()

    return result


def remove_quoted_content(body: str) -> str:
    """
    Remove quoted content from a reply body.
    Removes everything from "-----Original Message-----" onwards.
    """
    lines = body.split('\n')
    cleaned_lines = []

    for line in lines:
        # Stop at "-----Original Message-----" or "Original Message"
        if '-----Original Message-----' in line or ('Original Message' in line and '-----' in line):
            break
        # Also stop at lines that look like quoted headers (starting with "> From:", "> To:", etc.)
        stripped = line.strip()
        if stripped.startswith('>'):
            # Pre-compile header check for better performance
            stripped_lower = stripped.lower()
            if (stripped_lower.startswith('> from:') or stripped_lower.startswith('> to:') or
                stripped_lower.startswith('> subject:') or stripped_lower.startswith('> date:') or
                stripped_lower.startswith('> sent:')):
                break

        cleaned_lines.append(line)

    # Remove trailing empty lines
    while cleaned_lines and cleaned_lines[-1].strip() == '':
        cleaned_lines.pop()

    return '\n'.join(cleaned_lines).strip()


def extract_quoted_email_from_reply(body: str, reply_headers: Dict) -> Optional[Dict]:
    """
    Extract the original email from quoted content in a reply.
    Looks for "-----Original Message-----" or quoted content with ">" markers.
    """
    lines = body.split('\n')
    quoted_lines = []
    original_headers = {}

    # Look for "-----Original Message-----" marker
    original_msg_start = None
    for i, line in enumerate(lines):
        if '-----Original Message-----' in line or 'Original Message' in line:
            original_msg_start = i
            break

    if original_msg_start is not None:
        # Extract headers from lines after "Original Message"
        i = original_msg_start + 1
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines at start
            if not stripped and not original_headers:
                i += 1
                continue

            # Check for header patterns (handle both with and without ">" prefix)
            line_to_parse = line
            if stripped.startswith('>'):
                line_to_parse = line[1:].strip()  # Remove ">" prefix

            if ':' in line_to_parse:
                parts = line_to_parse.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()

                    # Clean up value using regex for email fields
                    key_lower = key.lower()
                    if key_lower in ['from', 'to', 'cc']:
                        # Use regex to extract email address
                        value = clean_email_field(value)
                    elif key_lower == 'subject':
                        # Clean subject (remove extra whitespace, but keep content)
                        value = value.strip()

                    # Store the cleaned value
                    if key_lower in ['from', 'to', 'subject', 'date', 'sent', 'cc']:
                        original_headers[key] = value

            # Empty line after headers indicates body start
            elif stripped == '' and original_headers:
                body_start = i + 1
                # Extract body (everything after headers until next email or end)
                body_lines = []
                for j in range(body_start, len(lines)):
                    body_line = lines[j]
                    # Remove ">" prefix if present
                    if body_line.strip().startswith('>'):
                        unquoted = body_line[1:] if body_line.startswith('>') else body_line
                        body_lines.append(unquoted)
                    elif body_line.strip() == '' and body_lines:
                        # Check if this is the end or start of another email
                        if j + 1 < len(lines) and ('-----Original Message-----' in lines[j+1] or
                                                   'Forwarded by' in lines[j+1]):
                            break
                        body_lines.append(body_line)
                    else:
                        body_lines.append(body_line)

                original_body = '\n'.join(body_lines).strip()

                return {
                    'headers': original_headers,
                    'body': original_body
                }

            i += 1

    # Fallback: Look for quoted content with ">" markers
    in_quoted = False
    quoted_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('>') and not stripped.startswith('>>'):
            in_quoted = True
            # Remove ">" prefix
            unquoted = line[1:] if line.startswith('>') else line
            quoted_lines.append(unquoted)
        elif in_quoted:
            if stripped == '' or stripped.startswith('>'):
                if stripped.startswith('>'):
                    quoted_lines.append(line[1:] if line.startswith('>') else line)
            else:
                break

    if quoted_lines:
        quoted_text = '\n'.join(quoted_lines)

        # Try to extract headers from quoted text using regex
        has_original_msg = 'Original Message' in quoted_text

        from_match = re.search(r'From:\s*(.+)', quoted_text, re.IGNORECASE | re.MULTILINE)
        if from_match and 'From' not in original_headers:
            original_headers['From'] = clean_email_field(from_match.group(1).strip())

        if has_original_msg:
            orig_msg_idx = quoted_text.find('Original Message')
            if orig_msg_idx >= 0:
                to_match = re.search(r'To:\s*(.+)', quoted_text[orig_msg_idx:], re.IGNORECASE | re.MULTILINE)
                if to_match and 'To' not in original_headers:
                    original_headers['To'] = clean_email_field(to_match.group(1).strip())
        else:
            if 'To' not in original_headers:
                to_match = re.search(r'To:\s*(.+)', quoted_text, re.IGNORECASE | re.MULTILINE)
                if to_match:
                    original_headers['To'] = clean_email_field(to_match.group(1).strip())

        subject_match = re.search(r'Subject:\s*(.+)', quoted_text, re.IGNORECASE | re.MULTILINE)
        if subject_match:
            original_headers['Subject'] = subject_match.group(1).strip()

        date_match = re.search(r'(?:Date|Sent):\s*(.+)', quoted_text, re.IGNORECASE | re.MULTILINE)
        if date_match:
            original_headers['Date'] = date_match.group(1).strip()

        reply_subject = reply_headers.get('Subject', '')
        if reply_subject.startswith('Re:') or reply_subject.startswith('RE:'):
            original_subject = re.sub(r'^Re:\s*', '', reply_subject, flags=re.IGNORECASE).strip()
            if original_subject and 'Subject' not in original_headers:
                original_headers['Subject'] = original_subject

        # Extract body (everything after headers)
        body_start = 0
        if original_headers:
            for i, line in enumerate(quoted_lines):
                if ':' in line and any(h in line for h in ['From', 'To', 'Subject', 'Date', 'Sent']):
                    continue
                elif line.strip() == '':
                    body_start = i + 1
                    break

        original_body = '\n'.join(quoted_lines[body_start:]).strip()

        # Infer some headers from the reply if not found
        if 'To' not in original_headers:
            original_headers['To'] = clean_email_field(reply_headers.get('From', ''))

        if 'From' not in original_headers:
            from_val = reply_headers.get('To', '')
            original_headers['From'] = clean_email_field(from_val)

        if original_body:
            return {
                'headers': original_headers,
                'body': original_body
            }

    return None


def parse_email_recursive(lines: List[str], start_idx: int, file_path: str,
                         depth: int = 0, max_depth: int = 10) -> Tuple[List[Dict], int]:
    """
    Recursively parse emails, handling forwards and replies.
    Returns (list_of_emails, next_index)
    """
    if depth >= max_depth:
        return [], start_idx

    emails = []
    i = start_idx

    # Skip empty lines at start
    while i < len(lines) and lines[i].strip() == '':
        i += 1

    if i >= len(lines):
        return [], i

    # Parse outer email headers
    current_headers, body_start = parse_email_headers(lines, i)

    if not current_headers:
        return [], i

    # Check if this is a forwarded email (has "Forwarded by" marker in body)
    has_forwarded_marker = False
    if body_start < len(lines):
        for j in range(body_start, min(body_start + 100, len(lines))):
            if 'Forwarded by' in lines[j]:
                has_forwarded_marker = True
                break
        if not has_forwarded_marker and body_start < len(lines):
            sample_text = '\n'.join(lines[body_start:body_start + 50])
            has_forwarded_marker = FORWARDED_PATTERN.search(sample_text) is not None

    # Extract body up to forwarded marker (if any)
    outer_body_end = len(lines)
    forwarded_marker_idx = None

    if has_forwarded_marker:
        for j in range(body_start, len(lines)):
            if 'Forwarded by' in lines[j] or (j > 0 and 'Forwarded by' in lines[j-1] and
                                               DASHES_PATTERN.search(lines[j])):
                forwarded_marker_idx = j
                outer_body_end = j - 1
                break

    outer_body = '\n'.join(lines[body_start:outer_body_end]).strip()

    # Create outer email
    subject = current_headers.get('Subject', '')
    is_forward = (subject.startswith('FW:') or subject.startswith('Fwd:') or
                  subject.startswith('Fw:') or has_forwarded_marker)

    outer_email = {
        'file': file_path,
        'email_index': len(emails),
        'is_outer': True,
        'is_forwarded': is_forward,
        'headers': current_headers,
        'body': outer_body,
        'depth': depth,
    }

    # Check if outer email is a reply
    subject = current_headers.get('Subject', '')
    is_reply = (subject.startswith('Re:') or subject.startswith('RE:') or
                subject.startswith('re:')) and '>' in outer_body

    if is_reply:
        outer_email['is_reply'] = True
        outer_email['body'] = remove_quoted_content(outer_body)
        original_email = extract_quoted_email_from_reply(outer_body, current_headers)
        if original_email:
            original_body = original_email['body']
            if 'Forwarded by' in original_body or '-----Original Message-----' in original_body:
                nested_lines = original_body.split('\n')
                nested_emails, _ = parse_email_recursive(nested_lines, 0, file_path, depth + 1, max_depth)
                if nested_emails:
                    emails.extend(nested_emails)
            else:
                emails.append({
                    'file': file_path,
                    'email_index': len(emails) + 1,
                    'is_outer': False,
                    'is_original': True,
                    'is_reply': True,
                    'headers': original_email['headers'],
                    'body': original_email['body'],
                    'depth': depth + 1,
                })

    emails.append(outer_email)

    # Process forwarded emails
    if forwarded_marker_idx is not None:
        i = forwarded_marker_idx
        while i < len(lines) and (DASHES_PATTERN.search(lines[i]) or lines[i].strip() == ''):
            i += 1

        if i < len(lines):
            if NESTED_FROM_PATTERN.match(lines[i].strip()):
                nested_headers = parse_nested_from_line(lines[i])
                nested_start = i
                i += 1

                while i < len(lines):
                    line = lines[i]
                    if ':' in line and not line.strip().startswith('>'):
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            key_lower = key.lower()
                            value = parts[1].strip()
                            if key_lower in ['from', 'to', 'cc']:
                                value = clean_email_field(value)
                            nested_headers[key] = value
                    elif line.strip() == '' and nested_headers:
                        nested_body_start = i + 1
                        break
                    i += 1
                else:
                    nested_body_start = i

                nested_body_end = len(lines)
                for k in range(nested_body_start, len(lines)):
                    if (EMAIL_HEADER_PATTERN.match(lines[k].strip()) or
                        NESTED_FROM_PATTERN.match(lines[k].strip()) or
                        'Forwarded by' in lines[k]) and k > nested_body_start + 5:
                        nested_body_end = k
                        break

                nested_body = '\n'.join(lines[nested_body_start:nested_body_end]).strip()

                nested_subject = nested_headers.get('Subject', '')
                is_nested_reply = (nested_subject.startswith('Re:') or nested_subject.startswith('RE:') or
                                  nested_subject.startswith('re:')) and '-----Original Message-----' in nested_body

                cleaned_nested_body = nested_body
                if is_nested_reply:
                    cleaned_nested_body = remove_quoted_content(nested_body)

                nested_email = {
                    'file': file_path,
                    'email_index': len(emails),
                    'is_outer': False,
                    'is_forwarded': False,
                    'is_reply': is_nested_reply,
                    'headers': nested_headers,
                    'body': cleaned_nested_body,
                    'depth': depth + 1,
                }

                nested_subject = nested_headers.get('Subject', '')
                is_nested_reply_for_extraction = (nested_subject.startswith('Re:') or nested_subject.startswith('RE:')) and '>' in nested_body

                if is_nested_reply_for_extraction:
                    original_email = extract_quoted_email_from_reply(nested_body, nested_headers)
                    if original_email:
                        original_body = original_email['body']
                        if 'Forwarded by' in original_body or '-----Original Message-----' in original_body:
                            nested_lines = original_body.split('\n')
                            deeper_emails, _ = parse_email_recursive(nested_lines, 0, file_path, depth + 2, max_depth)
                            if deeper_emails:
                                emails.append(nested_email)
                                emails.extend(deeper_emails)
                            else:
                                emails.append(nested_email)
                                emails.append({
                                    'file': file_path,
                                    'email_index': len(emails),
                                    'is_outer': False,
                                    'is_original': True,
                                    'is_reply': True,
                                    'headers': original_email['headers'],
                                    'body': original_email['body'],
                                    'depth': depth + 2,
                                })
                        else:
                            emails.append(nested_email)
                            emails.append({
                                'file': file_path,
                                'email_index': len(emails),
                                'is_outer': False,
                                'is_original': True,
                                'is_reply': True,
                                'headers': original_email['headers'],
                                'body': original_email['body'],
                                'depth': depth + 2,
                            })
                    else:
                        emails.append(nested_email)
                elif 'Forwarded by' in nested_body:
                    nested_lines = nested_body.split('\n')
                    deeper_emails, _ = parse_email_recursive(nested_lines, 0, file_path, depth + 1, max_depth)
                    if deeper_emails:
                        emails.append(nested_email)
                        emails.extend(deeper_emails)
                    else:
                        emails.append(nested_email)
                else:
                    emails.append(nested_email)

                return emails, nested_body_end

    return emails, len(lines)


def parse_email_message(file_path: str, message: str) -> List[Dict]:
    """
    Parse a single email message into a list of email objects.
    """
    lines = message.split('\n')
    emails, _ = parse_email_recursive(lines, 0, file_path, depth=0, max_depth=10)
    return emails


def parse_date(date_str: str) -> Optional[float]:
    """
    Parse date string to timestamp for sorting.
    Returns timestamp or None if parsing fails.
    """
    if not date_str:
        return None

    # Try common date formats
    date_formats = [
        '%a, %d %b %Y %H:%M:%S %z',      # Mon, 14 May 2001 16:39:00 -0700
        '%a, %d %b %Y %H:%M:%S',         # Mon, 14 May 2001 16:39:00
        '%m/%d/%Y %I:%M:%S %p',          # 02/16/2001 07:24:59 AM
        '%m/%d/%Y %H:%M:%S',             # 02/16/2001 07:24:59
        '%A, %B %d, %Y %I:%M %p',        # Friday, February 16, 2001 8:53 AM
        '%A, %B %d, %Y %I:%M:%S %p',     # Friday, February 16, 2001 8:53:00 AM
        '%B %d, %Y %I:%M %p',            # February 16, 2001 8:53 AM
    ]

    for fmt in date_formats:
        try:
            cleaned = date_str.strip()
            if 'PST' in cleaned or 'PDT' in cleaned:
                cleaned = re.sub(r'\s*\(PST\)', '', cleaned)
                cleaned = re.sub(r'\s*\(PDT\)', '', cleaned)

            dt = datetime.strptime(cleaned, fmt)
            return dt.timestamp()
        except Exception:
            continue

    return None


def convert_to_nested_json(file_path: str, message: str) -> Dict:
    """
    Convert an email message to nested JSON structure.
    Oldest email is outer, newer replies/forwards are nested.
    """
    emails = parse_email_message(file_path, message)

    if not emails:
        return None

    def get_sort_key(email):
        date_str = email['headers'].get('Date', '') or email['headers'].get('Sent', '')
        timestamp = parse_date(date_str)

        depth = email.get('depth', 0)
        is_original = email.get('is_original', False)

        if timestamp is None:
            if is_original:
                return (0, depth * -1, email.get('email_index', 0))
            return (1, depth * -1, email.get('email_index', 0))

        if is_original:
            return (timestamp - 100000000, email.get('email_index', 0))

        return (timestamp, email.get('email_index', 0))

    emails_sorted = sorted(emails, key=get_sort_key)

    oldest_email = emails_sorted[0]

    def build_nested_structure(email_list, index=0):
        """Recursively build nested structure from sorted email list."""
        if index >= len(email_list):
            return None

        current = email_list[index]
        nested = build_nested_structure(email_list, index + 1)

        result = {
            'headers': current['headers'],
            'body': current['body'],
            'is_forwarded': current.get('is_forwarded', False),
            'is_reply': current.get('is_reply', False),
            'is_original': current.get('is_original', False),
        }

        if nested:
            result['nested_emails'] = [nested]
        else:
            result['nested_emails'] = []

        return result

    structure = build_nested_structure(emails_sorted)

    if not structure:
        return None

    nested_emails = structure.get('nested_emails', [])

    result = {
        'file': file_path,
        'outer_email': {
            'headers': structure['headers'],
            'body': structure['body'],
            'is_forwarded': False,
            'is_reply': False,
        },
        'nested_emails': []
    }

    def flatten_nested(nested_email):
        """Recursively flatten nested emails into a list."""
        result_list = [{
            'headers': nested_email['headers'],
            'body': nested_email['body'],
            'is_forwarded': nested_email.get('is_forwarded', False),
            'is_reply': nested_email.get('is_reply', False),
            'is_original': nested_email.get('is_original', False),
        }]

        for nested in nested_email.get('nested_emails', []):
            result_list.extend(flatten_nested(nested))

        return result_list

    for nested in nested_emails:
        result['nested_emails'].extend(flatten_nested(nested))

    return result


def process_csv_to_json(input_csv: str, output_json: str):
    """
    Process CSV file and convert to nested JSON.
    """
    print(f"Processing {input_csv}...")
    print(f"Output: {output_json}\n")

    emails_list = []
    processed_count = 0
    filtered_count = 0
    error_count = 0
    total_processed = 0

    # Count total rows for progress bar
    print("Counting total emails...")
    with open(input_csv, 'r', encoding='utf-8') as fin:
        total_rows = sum(1 for _ in csv.DictReader(fin))
    print(f"Total emails to process: {total_rows:,}\n")

    with open(input_csv, 'r', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)

        pbar = tqdm(reader, total=total_rows, desc="Processing emails", unit="email")
        for row in pbar:
            total_processed += 1
            try:
                nested_json = convert_to_nested_json(row['file'], row['message'])
                if nested_json:
                    has_reply = (
                        nested_json['outer_email'].get('is_reply', False) or
                        any(nested.get('is_reply', False) for nested in nested_json.get('nested_emails', []))
                    )

                    if has_reply:
                        emails_list.append(nested_json)
                        processed_count += 1
                    else:
                        filtered_count += 1
                else:
                    error_count += 1
            except Exception as e:
                tqdm.write(f"Error processing {row['file']}: {e}")
                error_count += 1

            pbar.set_postfix({
                'valid': processed_count,
                'filtered': filtered_count,
                'errors': error_count
            })

    print(f"\nTotal emails processed: {total_processed}")
    print(f"Total emails with replies: {processed_count}")
    print(f"Filtered out (no replies): {filtered_count}")
    print(f"Errors: {error_count}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(emails_list, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_json) / (1024 * 1024)
    print(f"Saved to {output_json}")
    print(f"File size: {file_size:.2f} MB")

    emails_with_nested = sum(1 for e in emails_list if len(e.get('nested_emails', [])) > 0)
    max_nested = max((len(e.get('nested_emails', [])) for e in emails_list), default=0)

    print(f"\nStatistics:")
    print(f"  Emails with nested content: {emails_with_nested}")
    print(f"  Maximum nested emails per chain: {max_nested}")


def main():
    parser = argparse.ArgumentParser(description="Parse Enron emails from CSV and convert to nested JSON.")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file containing Enron emails.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    process_csv_to_json(args.input_csv, args.output_json)


if __name__ == '__main__':
    main()
