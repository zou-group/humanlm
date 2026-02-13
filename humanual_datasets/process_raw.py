import argparse
import asyncio
import gc
import hashlib
import importlib.util
import json
import os
import os.path as osp
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl
import rich
from datasets import (
    Dataset,
    get_dataset_config_names,
    get_dataset_infos,
    get_dataset_split_names,
    load_dataset,
)
from huggingface_hub import HfApi
from requests.exceptions import HTTPError
from rich.console import Group
from rich.live import Live
from rich.rule import Rule
from tqdm.rich import tqdm

from .utils_parser import (
    BRED,
    BYELLOW,
    RST,
    make_progress,
    memory_usage,
    parse_messages,
    parse_messages_title,
    patch_datasets_tqdm,
)
from .persona_generator import (
    UserPersonaGenerator,
    load_persona_checkpoint,
    save_persona_checkpoint,
)

patch_datasets_tqdm()

RANDOM_SEED = 42
MAX_ENTRIES_PER_SPLIT = 5_000_000


class RawDatasetProcessor:
    def __init__(
        self,
        dataset_name: str,
        splits: List[str],
        min_num_comments: int,
        max_num_comments: int,
        persona_history_length: int,
        pull_from_hub: str,
        llm_config: dict,
        push_to_hub: Optional[str] = None,
        save_dir: Optional[str] = None,
        unseen_frac: float = 0.1,
        seen_frac: float = 0.1,
        val_frac: float = 0.1,
        remove_used_persona_rows: bool = False,
        config_path: Optional[str] = None,
        verification_mode: str = "no_checks",
        subset_mode: bool = False,
        memory_friendly: bool = False,
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = False,
        compression: str = "uncompressed",
        partition_by: str = "user",
        preview_personas: bool = False,
        fixed_persona: Optional[str] = None,
        max_samples: Optional[int] = None,
        min_total_turns: int = 1,
        max_total_turns: int = 1,
        min_turns_for_train: int = 1,
        max_concurrent_users: int = 5,
        max_comment_length: int = 3000,
        truncate_comment_length: int = 1024,
        global_frac: float = 1.0,
        overwrite: bool = False,
    ):
        """
        memory_friendly: If True, we try to minimize the memory footprints, but it can slow things down meanwhile.
        max_samples: If provided, limit the total number of samples to process (for testing).
        """
        self.dataset_name = dataset_name
        self.pull_from_hub = pull_from_hub
        self.push_to_hub = push_to_hub
        self.save_dir = save_dir
        self.remove_used_persona_rows = remove_used_persona_rows
        self.llm_config = llm_config
        self.partition_by = partition_by
        self.max_concurrent_users = max_concurrent_users

        # make sure self.push_to_hub is not there or empty (unless overwrite is enabled)
        if self.push_to_hub is not None:
            if overwrite:
                # Delete existing repo contents to allow clean overwrite
                self._delete_repo_contents(self.push_to_hub)
            elif not self._is_repo_empty(self.push_to_hub):
                raise RuntimeError(f"Repo {self.push_to_hub} already exists and is not empty. Aborting. Use --overwrite to allow overwriting.")

        self.subset_mode = subset_mode
        self.raw_splits = splits
        # Here, `self.subsets` will be a list of subsets, each contains subset name and a list of splits, i.e., [(config_name, [split_names]), ...]
        self.subsets: list[tuple[str, Optional[list[str]]]] = self.parse_splits(splits)

        self.min_num_comments = min_num_comments
        self.max_num_comments = max_num_comments
        self.persona_history_length = persona_history_length
        self.unseen_frac = unseen_frac
        self.seen_frac = seen_frac
        self.val_frac = val_frac
        self.customized_filter_fn: Callable[[dict[str, Any]], bool] = self._load_customized_fn("customized_filter_fn")
        self.complete_dataset_fn: Callable[[pl.DataFrame, str], pl.DataFrame] = self._load_customized_fn("complete_dataset")

        self.config_path = config_path
        self.verification_mode = verification_mode
        self.memory_friendly = memory_friendly

        self.global_frac = global_frac

        # config_name, i.e., the subset name.
        maxs_suffix = f"-maxs{max_samples}" if max_samples is not None else ""
        gfrac_suffix = f"-gfrac{self.global_frac}" if self.global_frac < 1.0 else ""
        # min_max_config_name: used for caching the filtered dataset (after global_frac + min/max filtering)
        self.min_max_config_name = f"min{self.min_num_comments}-max{self.max_num_comments}{maxs_suffix}{gfrac_suffix}"
        # full_config_name: used for caching the final dataset (after persona generation and partitioning)
        self.full_config_name = f"min{self.min_num_comments}-max{self.max_num_comments}-hl{self.persona_history_length}{gfrac_suffix}"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.resume_from_cache = resume_from_cache
        self.compression = compression
        self.preview_personas = preview_personas
        self.fixed_persona = fixed_persona
        self.max_samples = max_samples
        self.min_total_turns = min_total_turns
        self.max_total_turns = max_total_turns
        self.min_turns_for_train = min_turns_for_train
        self.max_comment_length = max_comment_length
        self.truncate_comment_length = truncate_comment_length

        if self.cache_dir is not None:
            self.counts_cache_path = self.cache_dir / self.dataset_name / "user_comment_counts.json"
        else:
            self.counts_cache_path = None
        
        # Load dataset-specific configuration
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load YAML config for dataset."""
        import yaml
        config_path = Path(__file__).parent / "configs" / f"{self.dataset_name}.yaml"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _normalize_prompt(self, prompt):
        """Parse prompt if it's a JSON string."""
        if self.config["prompt_is_json_string"] and isinstance(prompt, str):
            return json.loads(prompt)
        return prompt

    def _normalize_metadata(self, metadata):
        """Normalize metadata to JSON string."""
        if self.config["metadata_is_json_string"] and isinstance(metadata, str):
            return metadata  # Already JSON string
        # It's a dict, stringify it
        return json.dumps(metadata) if metadata is not None else "{}"

    def _normalize_prompt_metadata(self, prompt_item_metadata):
        """Normalize metadata inside prompt items to JSON string."""
        if self.config["prompt_metadata_is_json_string"] and isinstance(prompt_item_metadata, str):
            return prompt_item_metadata  # Already JSON string
        # It's a dict, stringify it
        return json.dumps(prompt_item_metadata) if prompt_item_metadata else "{}"

    def _format_persona_context(self, row) -> str:
        """Format context for persona generation."""
        prompt = self._normalize_prompt(row["prompt"])
        if self.dataset_name == 'humanual_politics':
            context_text = parse_messages_title(prompt)
        else:
            context_text = parse_messages(prompt)
        
        # Add metadata enrichment if configured
        if "persona_enrichment" in self.config:
            enrichment = self.config["persona_enrichment"]
            if "use_metadata_fields" in enrichment:
                metadata = row["metadata"]
                if self.config["metadata_is_json_string"] and isinstance(metadata, str):
                    metadata = json.loads(metadata)
                
                enrichment_text = self._extract_enrichment_fields(metadata, enrichment["use_metadata_fields"])
                if enrichment_text:
                    context_text = f"{context_text}\n\n[Additional Context: {enrichment_text}]"
        
        return context_text

    def _extract_enrichment_fields(self, metadata: dict, field_paths: list) -> str:
        """Extract enrichment fields from metadata."""
        parts = []
        for path in field_paths:
            value = metadata
            for key in path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break
            if value is not None and value != "" and value != [] and value != 0:
                parts.append(f"{path}={value}")
        return ", ".join(parts)

    def _extract_dict(self, metadata: dict, field_paths) -> dict:
        """
        Extract selected fields from a nested metadata dict.

        takes either
            {
                "user.bio": ["bio"],
                "user.fullname": ["fullname"],
                "interests.tags_followed": ["Medium tags followed"],
            }
         or
            [
                {"user.bio": ["bio"]},
                {"user.fullname": ["fullname"]},
                {"interests.tags_followed": ["Medium tags followed"]},
            ]
        """
        result_inner = {}

        if isinstance(field_paths, dict):
            items = field_paths.items()
        else:
            items = []
            for mapping in field_paths:
                if isinstance(mapping, dict):
                    items.extend(mapping.items())
                else:
                    raise TypeError(f"Unexpected mapping type in field_paths: {type(mapping)}")

        for path, out_names in items:
            value = metadata
            ok = True
            for key in path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    ok = False
                    break

            # skip if missing
            if (not ok) or value in (None, "", []):
                continue

            for out_name in out_names:
                result_inner[out_name] = value

        return {
            "Medium user metadata": result_inner
        }


    def parse_splits(self, splits: list[str]) -> list[tuple[str, Optional[list[str]]]]:
        assert type(splits) is list and all(
            type(s) is str for s in splits
        ), f"`args.splits` must be a list of strings, got {splits}"
        print(f"Parsing splits: {splits}")
        print(f"Use split as config_name? {str(self.subset_mode)}")
        # Treat `splits` as regex patterns.
        if self.subset_mode:
            names = get_dataset_config_names(self.pull_from_hub)
        else:
            names = get_dataset_split_names(self.pull_from_hub)
        used_names = []
        for name in names:
            for pattern in splits:
                res = re.match(pattern, name)
                if res is not None:
                    used_names.append(name)
        print(f"Actually using splits: {used_names}")
        if len(used_names) == 0:
            raise ValueError(f"No split matched from {splits}, all available names: {names}")
        if self.subset_mode:
            return [(n, get_dataset_split_names(self.pull_from_hub, n)) for n in used_names]
        else:
            return [("default", used_names)]

    def _load_customized_fn(self, fn_name):
        """import `fn_name` from {dataset_name}.py."""
        module_path = Path(__file__).parent / f"{self.dataset_name}.py"
        spec = importlib.util.spec_from_file_location(f"humanual_datasets.{self.dataset_name}", str(module_path))
        if spec is None or spec.loader is None:
            print("Customized filter fn not found")
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, fn_name, None)
        return fn if callable(fn) else None

    def user_stat(self, ds: pl.DataFrame, num_bins=10):
        """Return and print user stats. Prints avg/min/max counts per user, number of unique users,
        historgram of counts per user, and number of users that would be dropped by persona_history_length
        or below min_num_comments."""
        bins = (
            list(np.linspace(0, self.min_num_comments, num_bins // 2, dtype=int))
            + list(np.linspace(self.min_num_comments, self.max_num_comments, num_bins // 2, dtype=int))
            + [10000]
        )
        if "user_id" in ds.columns and "count" in ds.columns:
            counts = ds.select(["user_id", "count"])
        else:
            counts = ds.group_by("user_id").agg(pl.len().alias("count")).sort("count", descending=True)

        n_users = int(counts.height)
        if n_users == 0:
            return {}

        count_s = counts["count"]
        unique_users = n_users
        avg_count = float(count_s.mean())
        min_count = int(count_s.min())
        max_count = int(count_s.max())

        edges = list(bins)
        labels = []
        conds = []
        for i in range(len(edges) - 1):
            left, right = edges[i], edges[i + 1]
            is_last = i == len(edges) - 2
            label = f"[{left}, {right}{']' if is_last else ')'}"
            labels.append(label)
            if is_last:
                cond = (pl.col("count") >= left) & (pl.col("count") <= right)
            else:
                cond = (pl.col("count") >= left) & (pl.col("count") < right)
            conds.append(cond)

        expr_label = None
        expr_ord = None
        for i, (label, cond) in enumerate(zip(labels, conds)):
            if expr_label is None:
                expr_label = pl.when(cond).then(pl.lit(label))
                expr_ord = pl.when(cond).then(pl.lit(i))
            else:
                expr_label = expr_label.when(cond).then(pl.lit(label))
                expr_ord = expr_ord.when(cond).then(pl.lit(i))

        expr_label = expr_label.otherwise(pl.lit("__out_of_range__"))
        expr_ord = expr_ord.otherwise(pl.lit(-1))

        bucketed = (
            counts.with_columns(
                [
                    expr_label.alias("bucket"),
                    expr_ord.alias("bucket_idx"),
                ]
            )
            .filter(pl.col("bucket_idx") >= 0)
            .group_by(["bucket", "bucket_idx"])
            .agg(pl.len().alias("users"))
            .sort("bucket_idx")
        )

        hist = {str(bucket): int(users) for bucket, users in zip(bucketed["bucket"].to_list(), bucketed["users"].to_list())}

        H = int(self.persona_history_length)
        if H <= 0:
            users_drop_entire = 0
            counts_drop_total = 0
        else:
            users_drop_entire = int(counts.filter(pl.col("count") <= H).height)
            counts_drop_total = int(
                counts.select(pl.min_horizontal(pl.col("count"), pl.lit(H)).sum().alias("sum")).to_series().item()
            )

        out: Dict[str, Any] = {
            "unique_users": unique_users,
            "avg_count_per_user": avg_count,
            "min_count_per_user": min_count,
            "max_count_per_user": max_count,
            "hist_count_per_user": hist,
            "persona_history_drop": {
                "history_len": H,
                "users_dropped_entirely": users_drop_entire,
                "counts_dropped_total": counts_drop_total,
            },
        }
        rich.print("User Stats Summary")
        rich.print({k: v for k, v in out.items() if k != "hist_count_per_user"})
        rich.print("Histogram (counts/user → users):")
        for b, u in out["hist_count_per_user"].items():
            rich.print(f"  {b}: {u}")

        return out

    def _is_repo_empty(self, repo_id: str) -> bool:
        """Return True if repo does not exist or exists but has no files."""
        api = HfApi()
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            return len(files) == 0
        except HTTPError as e:
            pass

        try:
            info = get_dataset_infos(repo_id)
            return len(info) == 0
        except Exception as e:
            return True

    def _delete_repo_contents(self, repo_id: str) -> None:
        """Delete all files in the repository to allow clean overwrite."""
        api = HfApi()
        try:
            files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            if files:
                print(f"[{datetime.now()}] Deleting {len(files)} existing files from {repo_id}...")
                # Delete files in batches to be more efficient
                batch_size = 50
                for i in range(0, len(files), batch_size):
                    batch = files[i:i + batch_size]
                    for file_path in batch:
                        try:
                            api.delete_file(
                                path_in_repo=file_path,
                                repo_id=repo_id,
                                repo_type="dataset",
                                commit_message=f"Delete files for overwrite (batch {i//batch_size + 1})" if i == 0 or i % batch_size == 0 else None
                            )
                        except Exception as e:
                            print(f"  Warning: Failed to delete {file_path}: {e}")
                print(f"[{datetime.now()}] Deleted all files from {repo_id}")
        except HTTPError as e:
            if e.response.status_code == 404:
                # Repo doesn't exist, nothing to delete
                print(f"[{datetime.now()}] Repo {repo_id} doesn't exist yet, nothing to delete")
            else:
                print(f"  Warning: Could not list files in {repo_id}: {e}")
        except Exception as e:
            print(f"  Warning: Error deleting repo contents: {e}")

    async def generate_personas_and_filter_history(self, ds: pl.DataFrame, train_only_ds: Optional[pl.DataFrame] = None, checkpoint_path: Optional[str] = None) -> pl.DataFrame:
        """
        Generate user personas and filter history.
        
        Args:
            ds: The full dataset to apply personas to
            train_only_ds: If provided, generate personas ONLY from this dataset (to avoid leakage)
            checkpoint_path: Path to save/load persona checkpoint (optional)
        """
        print(f"[{datetime.now()}] Generating user personas using their first {self.persona_history_length} comments...")
        
        # Load checkpoint if exists
        persona_batch_size = 100
        checkpoint_df, processed_users = load_persona_checkpoint(checkpoint_path)
        
        # Cost tracking
        persona_cost_tracker = {
            "total_cost_usd": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "num_calls": 0
        }
        
        # Use train_only_ds for persona generation if provided, otherwise use full ds
        persona_source_ds = train_only_ds if train_only_ds is not None else ds
        if train_only_ds is not None:
            print(f"  Using ONLY training data ({len(train_only_ds)} rows) for persona generation to avoid leakage")
        
        loop = asyncio.get_running_loop()  # TODO: Using coroutine sometimes doesn't speed up things. Maybe change to threads.
        persona_gen = UserPersonaGenerator(self.config["app_name"], max_concurrent_users=self.max_concurrent_users, config_path=self.config_path)
        user_data: list[dict] = []
        
        # Partition by user for persona generation
        user_to_rows_persona = persona_source_ds.partition_by("user_id", include_key=True, as_dict=True)
        
        # Partition by user for filtering (use full dataset)
        user_to_rows = ds.partition_by("user_id", include_key=True, as_dict=True)
        filtered_rows_list = []

        if self.persona_history_length == 0:
            print("Note: although persona_history_length=0, we still do traversing to check validity.")

        for (user_id,), rows in tqdm(
            user_to_rows.items(), desc="Filter history from all data", total=len(user_to_rows)
        ):
            rows = rows.sort("timestamp")
            if self.remove_used_persona_rows:
                if rows.height <= self.persona_history_length:
                    # Then we drop this user. NOTE: This is just defensive programming, beacause such users
                    # should have been dropped by `filter_by_num_comments` if called first.
                    continue
                non_history_user_comments = rows.tail(len(rows) - self.persona_history_length)
            else:
                non_history_user_comments = rows
            filtered_rows_list.append(non_history_user_comments)

        # Generate personas from persona_source_ds only
        for (user_id,), rows in tqdm(
            user_to_rows_persona.items(), desc="Generate user personas", total=len(user_to_rows_persona)
        ):
            # Skip if already processed
            if user_id in processed_users:
                continue
                
            rows = rows.sort("timestamp")
            history_rows = rows.head(min(self.persona_history_length, len(rows)))

            if "persona_enrichment" in self.config:
                # each row has same persona metadata so take first
                meta = history_rows["metadata"][0] 
                if self.config["metadata_is_json_string"] and isinstance(meta, str):
                    meta = json.loads(meta)
                user_meta_raw = meta.get("user_metadata", {})
                user_metadata = self._extract_dict(
                    user_meta_raw,
                    self.config["persona_enrichment"]["use_user_profile_fields"],
                )
            else:
                user_metadata = None

            if self.persona_history_length == 0:
                # No persona generation needed.
                user_data.append(
                    {
                        "user_id": user_id,
                        "persona": "Preview mode - no LLM call made" if self.fixed_persona is None else self.fixed_persona,
                    }
                )
            else:
                user_comment_texts = []
                for row in history_rows.iter_rows(named=True):
                    user_comment_texts.append(
                        {
                            "post_id": row["post_id"],
                            "timestamp": row["timestamp"],
                            "context": self._format_persona_context(row),
                            "comment_text": row["completion"],
                        }
                    )
                user_history = {"user_id": user_id, "comments": user_comment_texts}
                history_text = ""
                for i, comment in enumerate(user_history["comments"]):
                    ts = comment["timestamp"]
                    # Handle timestamps in milliseconds (convert to seconds)
                    if ts > 1e10:
                        ts = ts / 1000
                    readable_time = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
                    
                    # Truncate comment text if needed
                    comment_text = comment['comment_text']
                    if self.truncate_comment_length > 0:
                        words = comment_text.split()
                        if len(words) > self.truncate_comment_length:
                            comment_text = " ".join(words[:self.truncate_comment_length])
                    
                    text = f"<|Start of Example|>\nTimestamp: {readable_time}\n<|Start of the Context|>\n{comment['context']}\n<|End of the Context|>\n\n<|Start of the Target User's Response|>\n{comment_text}\n<|End of the Target User's Response|>\n\n<|End of Example|>\n\n"
                    
                    if (text_len := len(text.split())) > self.max_comment_length:
                        print(f"Warning: User {user_id} comment {i} is too long ({text_len} tokens), skipping...")
                        continue
                    history_text += text

                if self.preview_personas:
                    user_data.append(
                        {
                            "user_id": user_id,
                            "persona": "Preview mode - no LLM call made" if self.fixed_persona is None else self.fixed_persona,
                            "persona_preview_metadata": {
                                "num_comments_used": len(user_comment_texts),
                                "history_text": history_text,
                                "comments_used": user_comment_texts
                            }
                        }
                    )
                else:
                    user_data.append(
                        {
                            "user_id": user_id,
                            "persona": loop.create_task(persona_gen.generate_persona(user_id, history_text, self.config.get('use_user_profile_fields', None), user_metadata, **self.llm_config))
                        }
                    )
                    await asyncio.sleep(0.2)     

            # Save checkpoint after each batch
            if checkpoint_path is not None and len(user_data) > 0 and len(user_data) % persona_batch_size == 0:
                # print("Example persona", user_data[-1]["persona"])
                # Await any pending tasks in this batch
                if not self.preview_personas and self.persona_history_length != 0:
                    batch_start = len(user_data) - persona_batch_size
                    for i in range(batch_start, len(user_data)):
                        if isinstance(user_data[i]["persona"], asyncio.Task):
                            persona_result, cost_info = await user_data[i]["persona"]
                            user_data[i]["persona"] = persona_result
                            # Track costs
                            persona_cost_tracker["total_cost_usd"] += cost_info.get("cost_usd", 0.0)
                            persona_cost_tracker["total_prompt_tokens"] += cost_info.get("prompt_tokens", 0)
                            persona_cost_tracker["total_completion_tokens"] += cost_info.get("completion_tokens", 0)
                            persona_cost_tracker["total_tokens"] += cost_info.get("total_tokens", 0)
                            persona_cost_tracker["num_calls"] += 1
                
                # Convert to DataFrame and save checkpoint
                try:
                    batch_df = pl.DataFrame(user_data[-persona_batch_size:])
                except Exception:
                    batch_df = pl.DataFrame(
                        user_data[-persona_batch_size:],
                        schema_overrides={
                            "demographics": pl.Struct,
                            "interests": pl.List(pl.Utf8),
                            "values": pl.List(pl.Utf8),
                            "communication": pl.List(pl.Utf8),
                            "statistics": pl.List(pl.Utf8),
                        },
                    )

                checkpoint_df = save_persona_checkpoint(checkpoint_path, batch_df, checkpoint_df, processed_users)
        
        if self.persona_history_length != 0:
            if self.preview_personas:
                print(f"Preview mode: Generated preview metadata for {len(user_data)} users (no LLM calls made)")
            else:
                for user_datum in tqdm(user_data, desc="Await generating user personas", total=len(user_data)):
                    if isinstance(user_datum["persona"], asyncio.Task):
                        persona_result, cost_info = await user_datum["persona"]
                        user_datum["persona"] = persona_result
                        # Track costs
                        persona_cost_tracker["total_cost_usd"] += cost_info.get("cost_usd", 0.0)
                        persona_cost_tracker["total_prompt_tokens"] += cost_info.get("prompt_tokens", 0)
                        persona_cost_tracker["total_completion_tokens"] += cost_info.get("completion_tokens", 0)
                        persona_cost_tracker["total_tokens"] += cost_info.get("total_tokens", 0)
                        persona_cost_tracker["num_calls"] += 1

        # Save final batch if checkpointing
        if checkpoint_path is not None and len(user_data) > 0:
            # Find remaining users not yet saved
            if checkpoint_df is not None:
                saved_user_ids = set(checkpoint_df["user_id"].to_list())
                remaining_user_data = [ud for ud in user_data if ud["user_id"] not in saved_user_ids]
            else:
                remaining_user_data = user_data
            
            if remaining_user_data:
                # Await any pending tasks
                if not self.preview_personas and self.persona_history_length != 0:
                    for ud in remaining_user_data:
                        if isinstance(ud["persona"], asyncio.Task):
                            persona_result, cost_info = await ud["persona"]
                            ud["persona"] = persona_result
                            # Track costs
                            persona_cost_tracker["total_cost_usd"] += cost_info.get("cost_usd", 0.0)
                            persona_cost_tracker["total_prompt_tokens"] += cost_info.get("prompt_tokens", 0)
                            persona_cost_tracker["total_completion_tokens"] += cost_info.get("completion_tokens", 0)
                            persona_cost_tracker["total_tokens"] += cost_info.get("total_tokens", 0)
                            persona_cost_tracker["num_calls"] += 1
                
                # Convert to DataFrame and save final checkpoint
                final_batch_df = pl.DataFrame(remaining_user_data)
                checkpoint_df = save_persona_checkpoint(checkpoint_path, final_batch_df, checkpoint_df, processed_users)
                print(f"[{datetime.now()}] Saved final checkpoint: {len(processed_users)} total users processed")
        
        # Save persona cost summary
        if persona_cost_tracker["num_calls"] > 0:
            cost_summary_path = os.path.join(os.path.dirname(checkpoint_path) if checkpoint_path else ".", "persona_costs.json")
            with open(cost_summary_path, "w") as f:
                json.dump(persona_cost_tracker, f, indent=2)
            print(f"[{datetime.now()}] Persona generation costs: ${persona_cost_tracker['total_cost_usd']:.4f} "
                  f"({persona_cost_tracker['total_tokens']:,} tokens, {persona_cost_tracker['num_calls']} calls)")
        
        # Merge checkpoint personas with new user_data
        if len(filtered_rows_list) == 0:
            raise ValueError(f"No user left after filtering with persona_history_length={self.persona_history_length}")

        print(f"[{datetime.now()}] Reconstructing dataset and sort...")
        
        # Convert user_data to DataFrame
        if len(user_data) == 0:
            # If no new user data, create empty DataFrame with same schema as checkpoint
            if checkpoint_df is not None:
                # Create empty DataFrame with same columns as checkpoint
                user_data_df = checkpoint_df.head(0)
            else:
                user_data_df = pl.DataFrame({"user_id": [], "persona": []})
        else:
            user_data_df = pl.DataFrame(user_data)
            # Convert persona to JSON string for consistency
            if "persona" in user_data_df.columns:
                user_data_df = user_data_df.with_columns(
                    pl.col("persona").map_elements(
                        lambda p: json.dumps(p) if isinstance(p, dict) else str(p),
                        return_dtype=pl.String
                    ).alias("persona")
                )
        
        # Merge with checkpoint if exists
        if checkpoint_df is not None:
            if len(user_data_df) > 0:
                # Combine checkpoint and new personas, preferring new over checkpoint
                all_personas_df = pl.concat([checkpoint_df, user_data_df], how="vertical")
                # Take unique by user_id, keeping last (new) entries
                user_data_df = all_personas_df.unique(subset=["user_id"], keep="last")
            else:
                # No new personas, just use checkpoint
                user_data_df = checkpoint_df
        
        filtered_df: pl.DataFrame = pl.concat(filtered_rows_list)
        filtered_df = filtered_df.join(user_data_df, on="user_id", how="left", suffix="").sort(["post_id", "user_id", "timestamp"])
        
        # Handle users who don't have personas (i.e., appear in val/test but not in train when using train_only_ds)
        users_without_persona = filtered_df.filter(pl.col("persona").is_null()).select("user_id").unique()
        if len(users_without_persona) > 0:
            print(f"  {BYELLOW}Warning{RST}: {len(users_without_persona)} users appear in this split but not in training data.")
            print(f"  Assigning default persona: 'No training data available for this user'")
            filtered_df = filtered_df.with_columns(
                pl.when(pl.col("persona").is_null())
                .then(pl.lit("No training data available for this user"))
                .otherwise(pl.col("persona"))
                .alias("persona")
            )
        
        print(f"[{datetime.now()}] Reconstructed!")
        print(f"[{datetime.now()}] After filtering out persona history, {len(filtered_df)} entries left.")
        return filtered_df

    def pre_count_user_comments(self) -> dict[str, int]:
        if self.pull_from_hub is None:
            raise ValueError("If raw_dataset is not provided, pull_from_hub must be provided for source raw dataset(s).")
        # Directly load from cache if available.
        if self.counts_cache_path is not None and self.counts_cache_path.exists():
            print(f"[{datetime.now()}] Found cached comment counts: {self.counts_cache_path}, loading ...")
            user_comment_counts = json.load(open(self.counts_cache_path, "r"))
            return user_comment_counts

        # Cannot find from cache, do the counting.
        print(f"Loading raw dataset from Hugging Face Hub: {self.pull_from_hub} ...")
        user_comment_counts: dict[str, int] = {}
        subsets_progress = make_progress("[bold]{task.description}")
        split_progress = make_progress("[bold]  {task.description}")
        row_progress = make_progress("[bold]    {task.description}")
        group = Group(Rule(style="#AAAAAA"), subsets_progress, split_progress, row_progress)
        live = Live(group)
        with live:
            subsets_task_id = subsets_progress.add_task("Precounting", total=len(self.subsets))
            for subset, splits in self.subsets:
                subsets_progress.update(subsets_task_id, description=f"Loading {subset} since {datetime.now()}")
                ds = load_dataset(self.pull_from_hub, name=subset, verification_mode=self.verification_mode)
                subsets_progress.update(subsets_task_id, description=f"Overall")
                split_task_id = split_progress.add_task(f"Counting subset {subset}", total=len(splits))
                live.refresh()
                for k in splits:
                    v = ds[k]
                    acc_task_id = row_progress.add_task(f"Accumulating split {k}", total=len(v))
                    for user_id in v["user_id"]:
                        user_comment_counts[user_id] = user_comment_counts.get(user_id, 0) + 1
                        row_progress.update(acc_task_id, advance=1)
                    row_progress.update(acc_task_id, visible=False)
                    split_progress.update(split_task_id, advance=1)
                    live.refresh()
                split_progress.update(split_task_id, visible=False)
                subsets_progress.update(subsets_task_id, advance=1)
            subsets_progress.update(subsets_task_id, visible=False)

        # Cache it if required.
        if self.counts_cache_path is not None:
            os.makedirs(osp.dirname(self.counts_cache_path), exist_ok=True)
            json.dump(user_comment_counts, open(self.counts_cache_path, "w"), indent=2)

        return user_comment_counts

    def load_raw_dataset(self, allowed_user_ids: Optional[set[str]]) -> pl.DataFrame:
        if self.pull_from_hub is None:
            raise ValueError("If raw_dataset is not provided, pull_from_hub must be provided for source raw dataset(s).")
        print(f"Loading raw dataset from Hugging Face Hub: {self.pull_from_hub} ...")
        # NOTE: Looks like polars is much faster, but keep others for future reference.
        if allowed_user_ids is not None:
            allowed_user_ids = pl.DataFrame({"user_id": list(allowed_user_ids)})
        # NOTE: `ds_list` can be either `Dataset`, `polars.DataFrame`, or `str` (cache path).
        # At the end of this function, we will convert everything to the return type.
        ds_list: list[tuple[str, Dataset | pl.DataFrame | str]] = []  # If str, it is the cache path
        subsets_progress = make_progress("[bold]{task.description}")
        split_progress = make_progress("[bold]  {task.description}")
        row_progress = make_progress("[bold]    {task.description}")
        group = Group(Rule(style="#AAAAAA"), subsets_progress, split_progress, row_progress)
        live = Live(group)
        with live:
            subsets_task_id = subsets_progress.add_task("Loading", total=len(self.subsets))
            for subset, splits in self.subsets:
                print()
                subsets_progress.update(subsets_task_id, description=f"Loading {subset} since {datetime.now()}")
                
                # Calculate how many samples to load per split if max_samples is set
                if self.max_samples is not None and allowed_user_ids is None:
                    samples_per_split = max(50, self.max_samples // len(self.subsets) // max(len(splits), 1))
                    print(f"[{datetime.now()}] max_samples set: will load ~{samples_per_split} rows per split to avoid downloading full dataset")
                    # Load each split individually with slicing to avoid downloading everything
                    ds = {}
                    for split_name in splits:
                        split_slice = f"{split_name}[:{samples_per_split}]"
                        print(f"[{datetime.now()}] Loading {subset}/{split_slice} ...")
                        ds[split_name] = load_dataset(
                            self.pull_from_hub, 
                            name=subset, 
                            split=split_slice,
                            verification_mode=self.verification_mode
                        )
                else:
                    ds: dict[str, Dataset] = load_dataset(self.pull_from_hub, name=subset, verification_mode=self.verification_mode)
                
                subsets_progress.update(subsets_task_id, description=f"Overall")
                split_task_id = split_progress.add_task(f"Loading subset {subset}", total=len(splits))
                for k in splits:
                    v = ds[k]
                    subset_split_name = f"{subset} - {k}"
                    # Load from cache if available.
                    if self.cache_dir is not None:
                        v_cache_path: Path = (
                            self.cache_dir / self.dataset_name / "raw" / subset / f"{k}.parquet"
                        )
                        os.makedirs(osp.dirname(v_cache_path), exist_ok=True)
                        if self.resume_from_cache and v_cache_path.exists():
                            print(f"[{datetime.now()}] Resuming from cached {v_cache_path} ...")
                            ds_list.append((subset_split_name, str(v_cache_path)))
                            split_progress.update(split_task_id, advance=1)
                            live.refresh()
                            continue
                    # If provided `allowed_user_ids`, do the filtering.
                    v: pl.DataFrame = v.to_polars()
                    if allowed_user_ids is not None:
                        print(f"[{datetime.now()}] Filtering split {subset}/{k}.")
                        to_polars_task_id = row_progress.add_task(f"To polars for {subset}/{k}", total=1)
                        print(f"[{datetime.now()}] Converted to polars for {subset}/{k}.")
                        row_progress.update(to_polars_task_id, advance=1)
                        join_task_id = row_progress.add_task(f"Join for {subset}/{k}", total=1)
                        live.refresh()
                        v = v.join(allowed_user_ids, on="user_id", how="inner", suffix="")
                        row_progress.update(join_task_id, advance=1, visible=False)
                        row_progress.update(to_polars_task_id, visible=False)
                        print(f"[{datetime.now()}] After filtering, {len(v)} rows left.")
                    # Add necessary columns for future transformation.
                    print(f"[{datetime.now()}] Adding meta columns...")
                    v = v.with_columns(pl.lit(subset).alias("__config_name__"), pl.lit(k).alias("__split_name__"))
                    print(f"[{datetime.now()}] Added.")
                    # Cache to disk or append to `ds_list`
                    if self.cache_dir is not None:
                        print(f"[{datetime.now()}] Caching to {v_cache_path} ...")
                        v.write_parquet(v_cache_path, compression=self.compression, statistics=False)
                        print(f"[{datetime.now()}] Cached to {v_cache_path}.")
                        ds_list.append((subset_split_name, v_cache_path))
                    else:
                        ds_list.append((subset_split_name, v))
                    split_progress.update(split_task_id, advance=1)
                    print(f"[{datetime.now()}] Loaded config {subset}, split {k} with {len(v)} rows, {memory_usage()}")
                    live.refresh()
                split_progress.update(split_task_id, visible=False)
                subsets_progress.update(subsets_task_id, advance=1)
            subsets_progress.update(subsets_task_id, visible=False)
        if self.memory_friendly:
            gc.collect()
        print(f"[{datetime.now()}] Concatenating DataFrames to Dataset ...")

        final_list: list[pl.DataFrame] = []
        with make_progress("[progress.description]{task.description}") as progress:
            task = progress.add_task("Load/Concatenating datasets", total=len(ds_list))
            for name, ds in ds_list:
                progress.update(task, description=name)
                if type(ds) is str or isinstance(ds, Path):
                    ds = pl.read_parquet(str(ds))
                elif type(ds) is pl.DataFrame:
                    ds = ds
                elif type(ds) is Dataset:
                    ds = ds.to_polars()
                else:
                    raise TypeError(f"Unsupported type {type(ds)}")
                final_list.append(ds)
                progress.advance(task)

        # Do the concat
        print(f"[{datetime.now()}] Concatenate everything ({len(final_list)}) ...")
        df = pl.concat(final_list, how="vertical")
        print(f"[{datetime.now()}] Concatenated.")

        # Apply max_samples limit if specified (for testing/preview)
        if self.max_samples is not None and len(df) > self.max_samples:
            print(f"[{datetime.now()}] Limiting to {self.max_samples} samples (from {len(df)}) for testing/preview mode.")
            df = df.head(self.max_samples)

        return df

    async def create_dataset(self, raw_dataset: Dataset | pl.DataFrame = None) -> pl.DataFrame:
        """
        raw_dataset: if provided, directly use it; otherwise, load from HuggingFace Hub.
                     Can be a HuggingFace Dataset or a polars DataFrame.
        """
        if self.remove_used_persona_rows:
            assert self.min_num_comments > self.persona_history_length, "min_num_comments should be > persona_history_length"

        if raw_dataset is None:
            # Skip expensive user counting if max_samples is set (preview mode)
            if self.max_samples is not None:
                print(f"[{datetime.now()}] max_samples set: skipping user counting, will sample directly from dataset")
                df = self.load_raw_dataset(allowed_user_ids=None)
            else:
                user_comment_counts = self.pre_count_user_comments()
                user_comment_counts_df = pl.DataFrame(
                    {"user_id": list(user_comment_counts.keys()), "count": list(user_comment_counts.values())}
                )
                self.user_stat(user_comment_counts_df)
                print(f"[{datetime.now()}] Filtering user with #comments in [{self.min_num_comments}, {self.max_num_comments}] ...")
                allowed_user_ids = {
                    user_id
                    for user_id, count in user_comment_counts.items()
                    if self.min_num_comments <= count <= self.max_num_comments
                }
                print(f"[{datetime.now()}] After filtering: {len(allowed_user_ids)} users left.")
                df = self.load_raw_dataset(allowed_user_ids)
            n_unique_users = df.select(pl.col("user_id")).n_unique()
            n_unique_posts = df.select(pl.col("post_id")).n_unique()
            print(f'n_unique_users {n_unique_users} | n_unique_posts {n_unique_posts}')
        elif isinstance(raw_dataset, pl.DataFrame):
            # Pre-loaded polars DataFrame: filter by min/max user comment counts
            df = raw_dataset
            user_comment_counts = df.group_by("user_id").agg(pl.len().alias("count"))
            n_unique_users = user_comment_counts.height
            n_unique_posts = df.select(pl.col("post_id")).n_unique()
            print(f'n_unique_users {n_unique_users} | n_unique_posts {n_unique_posts}')
            self.user_stat(user_comment_counts)
            print(f"[{datetime.now()}] Filtering users with #comments in [{self.min_num_comments}, {self.max_num_comments}] ...")
            allowed_users = user_comment_counts.filter(
                (pl.col("count") >= self.min_num_comments) & (pl.col("count") <= self.max_num_comments)
            ).select("user_id")
            print(f"[{datetime.now()}] After filtering: {allowed_users.height} users left.")
            df = df.join(allowed_users, on="user_id", how="inner")
            print(f"[{datetime.now()}] After filtering: {len(df)} entries left.")
        else:
            # HuggingFace Dataset
            df = raw_dataset.to_polars()
            allowed_user_ids = (
                df.group_by("user_id")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .filter((pl.col("count") >= self.min_num_comments) & (pl.col("count") <= self.max_num_comments))
                .select(pl.col("user_id"))
            )
            df = df.join(allowed_user_ids, on="user_id", how="inner", suffix="")

        # Filter with user provided `customized_filter_fn`
        if self.customized_filter_fn:
            print(f"[{datetime.now()}] Applying customized_filter_fn to provided raw_dataset ...")
            mask = pl.struct(df.columns).map_elements(lambda s: self.customized_filter_fn(dict(s)))
            df = df.filter(mask)

        # Make the metadata column a JSON string
        df = df.with_columns(
            pl.col("metadata").map_elements(self._normalize_metadata, return_dtype=pl.String)
        )

        # For each prompt (a list of dicts), dump its internal metadata dicts as JSON strings
        # Example: [{"content": <str>, "role": "user", "metadata": <dict>}, ...]
        df = df.with_columns(
            pl.col("prompt").map_elements(
                lambda prompts: [
                    {
                        "content": m["content"],
                        "role": m["role"],
                        "metadata": self._normalize_prompt_metadata(m.get("metadata", {}))
                    }
                    for m in self._normalize_prompt(prompts)
                ],
                return_dtype=pl.List(pl.Struct([
                    pl.Field("content", pl.String),
                    pl.Field("role", pl.String),
                    pl.Field("metadata", pl.String),
                ]))
            )
        )

        # If user provided a `complete_dataset` function, use it to populate metadata fields etc.
        if self.complete_dataset_fn:
            print(f"[{datetime.now()}] Applying complete_dataset function to populate metadata fields ...")
            df = self.complete_dataset_fn(df, self.pull_from_hub)

        # Remove columns for `complete_dataset`
        to_drop = {"__idx__", "__config_name__", "__split_name__"} & set(df.columns)
        print(f"[{datetime.now()}] Dropping helper columns {to_drop} ...")
        df = df.drop(*to_drop)

        # Add `turn_id` and `conv_id` fields
        print(f"[{datetime.now()}] Add `turn_id` and `conv_id` fields...")
        df = df.with_columns([pl.col("prompt").list.len().alias("turn_id")])

        return df

    def partition_by_user(self, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """
        Partition by user_id.
                     →→→→→→→→→ Users
                        1-SEEN_FRACTION      | UNSEEN_FRACTION
        ↓          |---------------------------+--------|
        ↓          |                           |        |
        ↓          |                           |        |
        sort       |                           |        |
        by         |                           |        |
        timestamp  |                           |        |
                   |                           |        |
                   |---------------------------+        |
                   |    VAL_FRACTION           |        |
                   |---------------------------+        |
                   |    SEEN_TEST_FRACTION     |        |
                   |------------------------------------|
        """
        print(f"[{datetime.now()}] Partitioning dataset...")
        unique_user_ids = df.select("user_id").unique().sort("user_id")
        print(f"[{datetime.now()}] Total valid users: {len(unique_user_ids)}, total comments: {len(df)}")

        # 1). Split the dataset into seen/unseen datasets (sort is for stable sampling)
        unseen_user_ids = set(unique_user_ids.sample(fraction=self.unseen_frac, seed=RANDOM_SEED).to_series())
        unseen_test_df = df.filter(pl.col("user_id").is_in(unseen_user_ids))
        seen_df = df.filter(~pl.col("user_id").is_in(unseen_user_ids))
        print(f"[{datetime.now()}] all ({len(df)}) => seen ({len(seen_df)}) + unseen ({len(unseen_test_df)})")
        if len(unseen_test_df) == 0:
            print(f"{BYELLOW}Warning{RST}: no enough users for unseen test set. Adjust arguments if needed.")

        # 2). For each seen responsor, split into train/val and seen_test datasets
        seen_df_per_user = seen_df.partition_by("user_id", maintain_order=True, as_dict=True, include_key=True)
        seen_test_df_per_user = []
        seen_train_val_df_per_user = []
        seen_test_n_zero_counts = 0  # NOTE: for warning only
        for (user_id,), df in seen_df_per_user.items():
            df = df.sort("timestamp")
            n = len(df)
            seen_test_n = int(round(n * self.seen_frac))
            if seen_test_n <= 0:
                seen_test_n_zero_counts += 1
            seen_test_df_per_user.append(df[n - seen_test_n :])
            seen_train_val_df_per_user.append(df[: n - seen_test_n])
        if seen_test_n_zero_counts > 0:
            print(
                f"{BYELLOW}Warning{RST}: {seen_test_n_zero_counts} users have no enough comments for seen test set. "
                f"Given reasonable {self.min_num_comments=}, {self.seen_frac=}, each user should have at least one row in seen test set. Adjust them if not."
            )
        seen_test_df: pl.DataFrame = pl.concat(seen_test_df_per_user)
        train_val_df: pl.DataFrame = pl.concat(seen_train_val_df_per_user)
        print(f"[{datetime.now()}] seen ({len(seen_df)}) => train_val ({len(train_val_df)}) + seen ({len(seen_test_df)})")

        print(f"[{datetime.now()}] Partitioning train and val...  ({memory_usage()})")
        train_val_df = train_val_df.sample(fraction=1.0, seed=RANDOM_SEED)
        n_train = int(round(len(train_val_df) * (1 - self.val_frac)))
        n_val = len(train_val_df) - n_train
        train_df = train_val_df.slice(0, n_train)
        val_df = train_val_df.slice(n_train, n_val)

        # Collect all datasets into a dict.
        df_dict: dict[str, pl.DataFrame] = {
            "train": train_df,
            "val": val_df,
            "seen_test": seen_test_df,
            "unseen_test": unseen_test_df,
        }

        for k in list(df_dict.keys()):
            SORT_BY = ["post_id", "user_id", "timestamp"]
            print(f"[{datetime.now()}] Sorting {k} by {SORT_BY}...")
            df_dict[k] = df_dict[k].sort(SORT_BY)

        rich.print(f"[{datetime.now()}] Summary:", {k: {"columns": v.columns, "rows": len(v)} for k, v in df_dict.items()})

        return df_dict

    def partition_by_post(self, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """
        Partition by post_id splitting temporally.
        Posts are sorted by time, then split into train/val/test.

            →→→→→→→→→→→→→→→→ Posts (sorted by time) →→→→→→→→→→→→→→→→
            TRAIN_FRACTION      | VAL_FRACTION | TEST_FRACTION
        |------------------------+--------------|---------------------|
        |                        |              |                     |
        |    EARLY POSTS         | MIDDLE POSTS |   LATEST POSTS      |
        |    (train)             | (val)        |   (test)            |
        |                        |              |                     |
        |                        |              |                     |
        |---------------------------------------|---------------------|
        """
        print(f"[{datetime.now()}] Partitioning dataset by post (TEMPORAL)...")
        
        # Get earliest timestamp for each post and sort by time
        post_times = (
            df.group_by("post_id")
            .agg(pl.col("timestamp").min().alias("post_start_time"))
            .sort("post_start_time")
        )
        
        n_total = len(post_times)
        print(f"[{datetime.now()}] Total valid posts: {n_total}, total comments: {len(df)}")
        
        # Calculate split sizes
        # Note: We only use unseen_frac and val_frac; seen_frac is ignored for post-based splitting
        n_test = int(round(n_total * self.unseen_frac))
        n_val = int(round(n_total * self.val_frac))
        n_train = n_total - n_test - n_val

        print(f"[{datetime.now()}] Splitting {n_total} posts temporally:")
        print(f"  - Train: {n_train} earliest posts ({100*n_train/n_total:.1f}%)")
        print(f"  - Val: {n_val} middle posts ({100*n_val/n_total:.1f}%)")
        print(f"  - Test: {n_test} latest posts ({100*n_test/n_total:.1f}%)")

        # Split posts temporally
        train_post_ids = set(post_times.slice(0, n_train)["post_id"].to_list())
        val_post_ids = set(post_times.slice(n_train, n_val)["post_id"].to_list())
        test_post_ids = set(post_times.slice(n_train + n_val, n_test)["post_id"].to_list())
        
        # Get time ranges for each split
        train_times = post_times.slice(0, n_train)["post_start_time"]
        val_times = post_times.slice(n_train, n_val)["post_start_time"]
        test_times = post_times.slice(n_train + n_val, n_test)["post_start_time"]
        
        if len(train_times) > 0:
            train_start = datetime.fromtimestamp(train_times.min()).strftime("%Y-%m-%d")
            train_end = datetime.fromtimestamp(train_times.max()).strftime("%Y-%m-%d")
            print(f"  - Train time range: {train_start} to {train_end}")
        
        if len(val_times) > 0:
            val_start = datetime.fromtimestamp(val_times.min()).strftime("%Y-%m-%d")
            val_end = datetime.fromtimestamp(val_times.max()).strftime("%Y-%m-%d")
            print(f"  - Val time range: {val_start} to {val_end}")
        
        if len(test_times) > 0:
            test_start = datetime.fromtimestamp(test_times.min()).strftime("%Y-%m-%d")
            test_end = datetime.fromtimestamp(test_times.max()).strftime("%Y-%m-%d")
            print(f"  - Test time range: {test_start} to {test_end}")
        
        # Filter dataframe by post sets
        train_df = df.filter(pl.col("post_id").is_in(train_post_ids))
        val_df = df.filter(pl.col("post_id").is_in(val_post_ids))
        test_df = df.filter(pl.col("post_id").is_in(test_post_ids))

        # Collect all datasets into a dict.
        df_dict: dict[str, pl.DataFrame] = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        for k in list(df_dict.keys()):
            SORT_BY = ["post_id", "user_id", "timestamp"]
            print(f"[{datetime.now()}] Sorting {k} by {SORT_BY}...")
            df_dict[k] = df_dict[k].sort(SORT_BY)

        rich.print(f"[{datetime.now()}] Summary:", {k: {"columns": v.columns, "rows": len(v)} for k, v in df_dict.items()})

        return df_dict

    def partition_by_turn(self, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """
        Partition by turn order in each post.
        For each post, turns are sorted by turn_id (length of prompt list), then split into train/val/unseen_test.

                →→→→→→→→→→→→→→→→ Turns (sorted by turn_id) →→→→→→→→→→→→→→→→
                TRAIN_FRACTION      | VAL_FRACTION | UNSEEN_TEST_FRACTION
                |------------------------+--------------|---------------------|
        post 1  |                        |              |                     |
        post 2  |    EARLY TURNS         | MIDDLE TURNS |   LATEST TURNS      |
                |    (train)             | (val)        |   (unseen_test)     |
        ...     |                        |              |                     |
                |                        |              |                     |
        post n  |---------------------------------------|---------------------|
        """
        print(f"[{datetime.now()}] Partitioning dataset by turn order (based on prompt length)...")
        
        # Sort by post_id and turn_id to ensure correct turn ordering
        df = df.sort(["post_id", "turn_id"])
        
        # Get unique posts
        unique_post_ids = df.select("post_id").unique()
        n_posts = len(unique_post_ids)
        print(f"[{datetime.now()}] Total posts: {n_posts}, total comments: {len(df)}")
        
        # Partition each post by turn order
        train_rows = []
        val_rows = []
        seen_test_rows = []
        unseen_test_rows = []
        
        posts = df.partition_by("post_id", maintain_order=True, as_dict=True, include_key=True)
        
        posts_skipped = 0
        
        for (post_id,), post_df in posts.items():
            post_df = post_df.sort("turn_id")
            
            # Get unique turn_ids for this post
            unique_turn_ids = sorted(post_df["turn_id"].unique().to_list())
            n_unique_turns = len(unique_turn_ids)

            if n_unique_turns < self.min_total_turns or n_unique_turns > self.max_total_turns:
                posts_skipped += 1
                continue
            
            assert set(unique_turn_ids) == set(range(2, max(unique_turn_ids) + 1, 2)), f"turn_ids are not consecutive even numbers starting from 2 for post_id {post_id}"
            
            # Calculate split indices based on unique turns
            n_unseen_test_turns = int(np.ceil(n_unique_turns * self.unseen_frac)) + 1
            n_val_turns = max(int(n_unique_turns * self.val_frac) if self.val_frac > 0 else 0, 1)
            n_train_turns = max(self.min_turns_for_train, n_unique_turns - n_unseen_test_turns - n_val_turns)
            
            # Skip posts that don't have enough turns for training
            if n_train_turns < 1:
                posts_skipped += 1
                continue
            
            # Determine turn_id thresholds
            train_turn_ids = set(unique_turn_ids[:n_train_turns])
            val_turn_ids = set(unique_turn_ids[n_train_turns:n_train_turns + n_val_turns])
            test_turn_ids = set(unique_turn_ids[n_train_turns + n_val_turns:])
            
            # Split rows based on turn_id membership
            train_rows.append(post_df.filter(pl.col("turn_id").is_in(train_turn_ids)))
            if len(val_turn_ids) > 0:
                val_rows.append(post_df.filter(pl.col("turn_id").is_in(val_turn_ids)))
            if len(test_turn_ids) > 0:
                test_slice = post_df.filter(pl.col("turn_id").is_in(test_turn_ids))
                seen_test_rows.append(test_slice)
                unseen_test_rows.append(test_slice)
        
        if posts_skipped > 0:
            print(f"{BYELLOW}Warning{RST}: {posts_skipped} posts skipped due to insufficient turns for splits.")
        
        # Concatenate all splits
        if len(train_rows) == 0:
            raise ValueError("No posts have enough turns for splits. Consider lowering min_total_turns or adjusting split fractions.")
        
        train_df = pl.concat(train_rows)
        val_df = pl.concat(val_rows) if val_rows else train_df.head(0)  # Empty df with same schema
        seen_test_df = pl.concat(seen_test_rows) if seen_test_rows else train_df.head(0)
        unseen_test_df = pl.concat(unseen_test_rows) if unseen_test_rows else train_df.head(0)
        
        print(f"[{datetime.now()}] all ({len(df)}) => train ({len(train_df)}) + val ({len(val_df)}) + seen_test ({len(seen_test_df)}) + unseen_test ({len(unseen_test_df)})")
        
        # Collect all datasets into a dict
        df_dict: dict[str, pl.DataFrame] = {
            "train": train_df,
            "val": val_df,
            "seen_test": seen_test_df,
            "unseen_test": unseen_test_df,
        }
        
        for k in list(df_dict.keys()):
            SORT_BY = ["post_id", "user_id", "timestamp"]
            print(f"[{datetime.now()}] Sorting {k} by {SORT_BY}...")
            df_dict[k] = df_dict[k].sort(SORT_BY)
        
        rich.print(f"[{datetime.now()}] Summary:", {k: {"columns": v.columns, "rows": len(v)} for k, v in df_dict.items()})
        
        return df_dict

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process Raw Dataset.")
    ###### Dataset Properties ######
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["humanual_book", "humanual_opinion", "humanual_news", "humanual_politics", "humanual_email", "humanual_chat"],
        help="The dataset name: humanual_book|humanual_opinion|humanual_news|humanual_politics|humanual_email|humanual_chat",
    )
    parser.add_argument("--splits", nargs="+", type=str, required=True, help="A list of split names of raw dataset to include.")
    # Args for filtering
    parser.add_argument(
        "--max_num_comments",
        type=int,
        default=1000,
        help="Filter out users with reviews more than this number",
    )
    parser.add_argument(
        "--min_num_comments",
        type=int,
        default=10,
        help="Filter out users with reviews less than this number",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit the total number of samples to process (for testing/preview)",
    )
    # Args for persona
    parser.add_argument(
        "--persona_history_length",
        type=int,
        default=20,
        help="Number of reviews to use for generating user persona. It should be < min_num_comments. If 0, no persona will be generated.",
    )
    parser.add_argument(
        "--remove_used_persona_rows",
        action="store_true",
        help="If set, remove the rows that are used for persona generation from the final dataset.",
    )
    # Args for partitioning (See details and visualization in `partition` method)
    parser.add_argument(
        "--unseen_frac",
        type=float,
        default=0.08,
        help="Fraction of users data to hold out as unseen test set",
    )
    parser.add_argument(
        "--seen_frac",
        type=float,
        default=0.08,
        help="Fraction of comment data from seen users to hold out as seen test set",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.02,
        help="Fraction of comment data from seen users to hold out as validation set",
    )
    ###### Env args ######
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to .env file with necessary credentials (e.g., OpenAI API key)",
    )
    ###### Download/Upload/Saving ######
    parser.add_argument(
        "--pull_from_hub",
        type=str,
        default=None,
        help="Push raw dataset from Hugging Face Hub",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="Push dataset to Hugging Face Hub (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing HuggingFace Hub repository",
    )
    parser.add_argument("--save_dir", type=str, default=None, help="Store parquet files to local dir")
    ###### Optimization ######
    parser.add_argument(
        "--verification_mode",
        type=str,
        default="no_checks",
        choices=["no_checks", "basic_checks", "all_checks"],
        help="Loading data from hub might fail due to some stupid issues; you can try to set this to 'no_checks' to skip all verification.",
    )
    parser.add_argument(
        "--subset_mode",
        action="store_true",
        help="We later realized that datasets should use subset for different channels/categories, instead of splits.",
    )
    parser.add_argument(
        "--memory_friendly",
        action="store_true",
        help="If True, we try to minimize the memory footprints, but it can slow things down meanwhile.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="If provided, we cache intermediate datasets to this local dir to save memory (optional).",
    )
    parser.add_argument(
        "--cache_hub",
        type=str,
        default=None,
        help="If provided, we load the whole dataset (before persona generation and partitioning) from this hub repo; but if `cache_dir` is also provided, we will still load from local cache.",
    )
    parser.add_argument(
        "--resume_from_cache",
        action="store_true",
        help="If True, we will resume from existing cache dir if possible (optional).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="uncompressed",
        choices=["uncompressed", "snappy", "lz4", "zstd"],
        help="Compression algorithm to use for cache.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="anthropic/claude-sonnet-4-5-20250929",
        help="LLM model to use for persona generation",
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=0.0,
        help="LLM temperature to use for persona generation",
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=4096,
        help="LLM max tokens to use for persona generation",
    )
    parser.add_argument(
        "--partition_by",
        type=str,
        default="user",
        choices=["user", "post", "turn"],
        help="Partition by user or by post or by turns",
    )
    parser.add_argument(
        "--min_turns_for_train",
        type=int,
        default=0,
        help="Minimum number of turns",
    )
    parser.add_argument(
        "--min_total_turns",
        type=int,
        default=0,
        help="Minimum number of turns in for the prompt conversation",
    )
    parser.add_argument(
        "--max_total_turns",
        type=int,
        default=0,
        help="Maximum number of turns in for the prompt conversation",
    )
    parser.add_argument(
        "--preview_personas",
        action="store_true",
        help="Preview mode: show comments/posts that would be used for persona generation without making LLM calls",
    )
    parser.add_argument(
        "--fixed_persona",
        type=str,
        default=None, 
        help="A string of persona to use for all users ",
    )
    parser.add_argument(
        "--max_comment_length",
        type=int,
        default=3000,
        help="Comments with a length longer than this will be removed for persona generation",
    )
    parser.add_argument(
        "--truncate_comment_length",
        type=int,
        default=1024,
        help="Truncate comments to this word length (in words) for persona generation. Set to 0 to disable truncation.",
    )
    parser.add_argument(
        "--max_concurrent_users",
        type=int,
        default=5,
        help="Number of concurrent LLM calls for persona generation",
    )
    parser.add_argument(
        "--global_frac",
        type=float,
        default=1.0,
        help="Randomly sample this fraction of the raw dataset before processing (default: 1.0, no sampling)",
    )


    args = parser.parse_args()

    # hash args to get a unique config name
    ############## Argument parsing and analyzing ##############

    # Only hash the configuration parameters that affect results
    config_keys = [
        'dataset_name', 'splits', 'max_num_comments', 'min_num_comments',
        'persona_history_length', 'remove_used_persona_rows',
        'unseen_frac', 'seen_frac', 'val_frac', 'partition_by',
        'llm_model', 'llm_temperature', 'llm_max_tokens', 'max_samples', 'max_comment_length',
        'global_frac',
    ]
    
    # Create a deterministic config dict with sorted keys
    config_dict = {k: getattr(args, k) for k in sorted(config_keys)}
    
    # Use JSON for deterministic string representation
    args_str = json.dumps(config_dict, sort_keys=True)
    hash_str = hashlib.md5(args_str.encode()).hexdigest()[:8]
    
    ########### Checkpoint paths ###########
    # Default checkpoint directories relative to project root
    project_root = Path(__file__).parent.parent.parent
    default_persona_checkpoint_dir = project_root / "persona_checkpoints"
    if args.cache_dir is not None:
        args.persona_checkpoint_path = osp.join(args.cache_dir, f"persona_checkpoints/{args.dataset_name}/{hash_str}/personas.parquet")
    else:
        args.persona_checkpoint_path = str(default_persona_checkpoint_dir / args.dataset_name / hash_str / "personas.parquet")

    print(f"[{datetime.now()}] Using checkpoint paths:")
    print(f"  - persona_checkpoint_path: {args.persona_checkpoint_path}")
    
    ############## Argument validation ##############
    if args.remove_used_persona_rows and args.min_num_comments < args.persona_history_length:
        raise ValueError(
            f"min_num_comments ({args.min_num_comments}) should be >= persona_history_length ({args.persona_history_length})"
        )
    assert 0 < args.global_frac <= 1.0, "global_frac should be in (0, 1]"
    assert 0 <= args.seen_frac < 1, "seen_frac should be in [0, 1)"
    assert 0 <= args.unseen_frac < 1, "unseen_frac should be in [0, 1)"
    assert 0 <= args.val_frac < 1, "val_frac should be in [0, 1)"
    assert 0 <= args.val_frac + args.seen_frac < 1, "val_frac + seen_frac should be in [0, 1)"
    if args.partition_by == "user" and args.min_num_comments * args.seen_frac < 1:
        print(
            f"{BYELLOW}Warning{RST}: Given min_num_comments={args.min_num_comments}, seen_frac={args.seen_frac},"
            f" each user will have less than one comment for seen test dataset. Adjust them if needed."
        )

    if args.resume_from_cache and args.cache_dir is None:
        raise ValueError("If resume_from_cache is enabled, cache_dir must be provided.")

    ############## End of argument parsing and analyzing ##############

    rdp = RawDatasetProcessor(
        dataset_name=args.dataset_name,
        pull_from_hub=args.pull_from_hub,
        push_to_hub=args.push_to_hub,
        save_dir=args.save_dir,
        splits=args.splits,
        min_num_comments=args.min_num_comments,
        max_num_comments=args.max_num_comments,
        persona_history_length=args.persona_history_length,
        remove_used_persona_rows=args.remove_used_persona_rows,
        config_path=args.config,
        verification_mode=args.verification_mode,
        subset_mode=args.subset_mode,
        memory_friendly=args.memory_friendly,
        cache_dir=args.cache_dir,
        resume_from_cache=args.resume_from_cache,
        compression=args.compression,
        unseen_frac=args.unseen_frac,
        seen_frac=args.seen_frac,
        val_frac=args.val_frac,
        partition_by=args.partition_by,
        llm_config={
            "model": args.llm_model,
            "temperature": args.llm_temperature,
            "max_tokens": args.llm_max_tokens,
        },
        preview_personas=args.preview_personas,
        fixed_persona=args.fixed_persona,
        max_samples=args.max_samples,
        min_total_turns=args.min_total_turns,
        max_total_turns=args.max_total_turns,
        min_turns_for_train=args.min_turns_for_train,
        max_concurrent_users=args.max_concurrent_users,
        max_comment_length=args.max_comment_length,
        truncate_comment_length=args.truncate_comment_length,
        global_frac=args.global_frac,
        overwrite=args.overwrite,
    )

    # Pipeline: load raw → apply global_frac → apply min/max filtering → persona generation → partitioning
    # Cache check order: filtered (most processed) → sampled → raw (least processed)

    df_cache_path = (
        Path(rdp.cache_dir) / rdp.dataset_name / rdp.min_max_config_name / "all_data.parquet" if rdp.cache_dir else None
    )
    sampled_cache_path = (
        Path(rdp.cache_dir) / rdp.dataset_name / f"all_data_gfrac{args.global_frac}.parquet"
        if rdp.cache_dir and args.global_frac < 1.0 else None
    )

    if rdp.resume_from_cache and df_cache_path is not None and df_cache_path.exists():
        # Fast path: filtered dataset already cached
        print(f"[{datetime.now()}] Resuming from filtered dataset cache at {df_cache_path} ...")
        df = pl.read_parquet(df_cache_path)
        print(f"[{datetime.now()}] Resumed with {len(df)} rows, {memory_usage()}")
    else:
        # Need to load or resume from an earlier stage, then filter
        if rdp.resume_from_cache and sampled_cache_path is not None and sampled_cache_path.exists():
            # Resume from sampled cache
            print(f"[{datetime.now()}] Resuming from sampled dataset cache at {sampled_cache_path} ...")
            df = pl.read_parquet(sampled_cache_path)
            print(f"[{datetime.now()}] Resumed sampled data with {len(df)} rows, {memory_usage()}")
        else:
            # Load raw dataset (per-split caching in load_raw_dataset handles HF Hub downloads)
            if args.cache_hub is not None:
                print(f"[{datetime.now()}] Loading dataset from hub cache: {args.cache_hub} ...")
                ds = load_dataset(args.cache_hub, split="train", verification_mode=args.verification_mode)
                print(f"[{datetime.now()}] Loaded data from hub (len={len(ds)}). Now converting to polars DataFrame ...")
                df = ds.to_polars()
                print(f"[{datetime.now()}] Converted to polars DataFrame")
            else:
                df = rdp.load_raw_dataset(allowed_user_ids=None)
                print(f"[{datetime.now()}] Total entries in the raw dataset: {len(df)}.")

            # Apply global_frac sampling and cache
            if args.global_frac < 1.0:
                original_len = len(df)
                df = df.sample(fraction=args.global_frac, seed=RANDOM_SEED)
                print(f"[{datetime.now()}] global_frac={args.global_frac}: sampled {len(df)} rows from {original_len}")
                if sampled_cache_path is not None:
                    os.makedirs(osp.dirname(sampled_cache_path), exist_ok=True)
                    print(f"[{datetime.now()}] Caching sampled dataset to {sampled_cache_path} ...")
                    df.write_parquet(sampled_cache_path, compression=rdp.compression, statistics=False)
                    print(f"[{datetime.now()}] Cached sampled dataset.")

        # Apply min/max filtering + normalization and cache
        df = asyncio.run(rdp.create_dataset(raw_dataset=df))
        print(f"[{datetime.now()}] Total entries after filtering: {len(df)}.")
        if df_cache_path is not None:
            os.makedirs(osp.dirname(df_cache_path), exist_ok=True)
            print(f"[{datetime.now()}] Caching the filtered dataset to {df_cache_path} ...")
            df.write_parquet(df_cache_path, compression=rdp.compression, statistics=False)
            print(f"[{datetime.now()}] Cached the filtered dataset to {df_cache_path}.")

    # ------- Any cache above should be saved to `rdp.min_max_config_name`
    # ------- Any cache below should be saved to `rdp.full_config_name`
    # Partition the dataset (NOTE: partition has to be done after persona generation because the history may be dropped.)
    if rdp.partition_by in ["user", "turn"]:
        # For user-based partitioning, we can generate personas before splitting (no leakage issue)
        print(f"[{datetime.now()}] User-based partitioning: generating personas before splitting...")
        
        # Use a single event loop for all async operations to avoid litellm worker conflicts
        async def generate_personas_with_cleanup():
            try:
                return await rdp.generate_personas_and_filter_history(df, checkpoint_path=args.persona_checkpoint_path)
            finally:
                # Cancel all remaining tasks to allow event loop to close cleanly
                loop = asyncio.get_running_loop()
                tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                # Wait for cancellation to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        df = asyncio.run(generate_personas_with_cleanup())
        
        if rdp.partition_by == "user":
            df_dict: dict[str, pl.DataFrame] = rdp.partition_by_user(df)
        elif rdp.partition_by == "turn":
            df_dict: dict[str, pl.DataFrame] = rdp.partition_by_turn(df)
        else:
            raise ValueError(f"Unknown partition_by: {rdp.partition_by}")

    elif rdp.partition_by == "post":
        # For post-based partitioning, we partition first to avoid leakage, then generate personas
        print(f"[{datetime.now()}] Post-based partitioning: splitting first to avoid leakage...")
        df_dict_before_persona: dict[str, pl.DataFrame] = rdp.partition_by_post(df)
        
        # Identify users who appear in training data (only these will have personas)
        train_df = df_dict_before_persona["train"]
        train_user_ids = set(train_df.select("user_id").unique().to_series().to_list())
        print(f"[{datetime.now()}] Found {len(train_user_ids)} unique users in training split")
        
        # Filter val/test splits to only include users who appear in training
        # This ensures all users in val/test have personas generated from their training data
        for split_name in ["val", "test"]:
            split_df = df_dict_before_persona[split_name]
            original_rows = len(split_df)
            original_users = split_df.select("user_id").n_unique()
            
            # Filter to only users who appear in training
            split_df = split_df.filter(pl.col("user_id").is_in(list(train_user_ids)))
            filtered_rows = len(split_df)
            filtered_users = split_df.select("user_id").n_unique()
            
            rows_removed = original_rows - filtered_rows
            users_removed = original_users - filtered_users
            
            if rows_removed > 0:
                print(f"  {BYELLOW}{split_name}{RST}: Removed {users_removed} users ({rows_removed} rows) who don't appear in training")
            
            df_dict_before_persona[split_name] = split_df

        # Generate personas for ALL users, but using ONLY training data for persona content
        print(f"[{datetime.now()}] Generating personas for all users using training split data only...")
        
        # Use a single event loop for all async operations to avoid litellm worker conflicts
        async def generate_personas_with_cleanup():
            try:
                return await rdp.generate_personas_and_filter_history(df, train_only_ds=train_df, checkpoint_path=args.persona_checkpoint_path)
            finally:
                # Cancel all remaining tasks to allow event loop to close cleanly
                loop = asyncio.get_running_loop()
                tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                # Wait for cancellation to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
        
        df_with_personas = asyncio.run(generate_personas_with_cleanup())

        # Apply personas to the already-partitioned splits
        print(f"[{datetime.now()}] Applying personas to partitioned splits...")
        persona_data = df_with_personas.select(["user_id", "persona"]).unique()

        # [NEW] bug fix
        comment_meta = df_with_personas.select([
            "post_id",
            "user_id",
            "timestamp",
            "prompt",    # now here
            "metadata",  # and here
        ])
        df_dict: dict[str, pl.DataFrame] = {}
        for split_name, split_df in df_dict_before_persona.items():
            # [NEW CHANGE]
            split_df = split_df.join(persona_data, on="user_id", how="left")
            split_df = split_df.drop(["prompt", "metadata"], strict=False)
            split_df = split_df.join(
                comment_meta,
                on=["post_id", "user_id", "timestamp"],
                how="left",
            )
            
            # Sanity check: After filtering, there should be no users without personas in val/test
            users_without_persona = split_df.filter(pl.col("persona").is_null()).select("user_id").n_unique()
            if users_without_persona > 0:
                if split_name == "train":
                    # This is expected for train if persona_history_length filtering removed some users
                    print(f"  {BYELLOW}Warning{RST}: {split_name} has {users_without_persona} users without personas (likely filtered by persona_history_length)")
                else:
                    # This should NOT happen for val/test after our filtering
                    print(f"  {BRED}Error{RST}: {split_name} has {users_without_persona} users without personas - this should not happen!")
                
                split_df = split_df.with_columns(
                    pl.when(pl.col("persona").is_null())
                    .then(pl.lit("No training data available for this user"))
                    .otherwise(pl.col("persona"))
                    .alias("persona")
                )
            
            df_dict[split_name] = split_df
    else:
        raise ValueError(f"Unknown partition_by method: {rdp.partition_by}")
    # ------- Saving or Uploading -------
    # NOTE: Directly converting polars DataFrame to HF Dataset could lead to OOM, hence we are doing:
    # 1. **Save-reload**: Saving `pl.DataFrame` as parquet files then reload with `Dataset.from_parquet`.
    #    This saves memory because writing parquet directly from polars DataFrame uses less memory.
    #    Besides `Dataset.from_parquet` loads parquet files without too much memory overhead.
    # 2. **Partition**: Partition each split into smaller pieces (e.g., `MAX_ENTRIES_PER_SPLIT` entries per piece).
    #    This helps to further reduce memory usage during conversion with Dataset.from_polars or Dataset.from_list.
    # Meanwhile, we release the memory as soon as it is saved (pushed to hub or save locally)
    # To enable `save-reload`, provide `rdp.cache_dir`.
    for split in list(df_dict.keys()):
        df: pl.DataFrame = df_dict[split]
        print(f"[{datetime.now()}] Converting {split} to HF Dataset: {len(df)} rows, {memory_usage()}")
        partition_points = list(range(0, len(df), MAX_ENTRIES_PER_SPLIT)) + [len(df)]

        for i in tqdm(range(len(partition_points) - 1), desc=f"Processing {split}", total=len(partition_points) - 1):
            part_name = f"{split}{i}"
            print(f"[{datetime.now()}] Working on {part_name}, rows {partition_points[i]} ~ {partition_points[i+1]} ...")
            df_part = df[partition_points[i] : partition_points[i + 1]]

            if rdp.cache_dir:
                # Save to local cache first.
                cache_path = Path(rdp.cache_dir) / rdp.dataset_name / rdp.full_config_name / f"{part_name}.parquet"
                os.makedirs(osp.dirname(cache_path), exist_ok=True)
                print(f"[{datetime.now()}] Saving {part_name} to {cache_path} ...")
                df_part.write_parquet(cache_path, compression=rdp.compression)
                print(f"[{datetime.now()}] Saved.")

            entries = [e for e in df_part.iter_rows(named=True)]
            ds_part = Dataset.from_list(entries)
            
            if rdp.push_to_hub:
                print(f"[{datetime.now()}] Pushing partition {part_name} to hub ...")
                ds_part.push_to_hub(
                    rdp.push_to_hub, 
                    config_name=rdp.full_config_name, 
                    split=part_name, 
                    private=True
                )

            if rdp.save_dir:
                os.makedirs(rdp.save_dir, exist_ok=True)
                out_path = os.path.join(rdp.save_dir, f"{part_name}.parquet")
                if rdp.cache_dir:
                    shutil.move(cache_path, out_path)
                    print(f"[{datetime.now()}] Moved {part_name}.parquet with {len(ds_part)} rows to {out_path}")
                else:
                    ds_part.to_parquet(out_path)
                    print(f"[{datetime.now()}] Saved {part_name}.parquet with {len(ds_part)} rows at {out_path}")
            if rdp.memory_friendly:
                del ds_part
        
        if rdp.memory_friendly:
            del df
            del df_dict[split]