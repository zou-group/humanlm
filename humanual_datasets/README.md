# Humanual Datasets

This package provides tools for collecting, processing, and partitioning datasets for human-language modeling. Each dataset describes a one-to-many relation between `post`s and `comment`s.

**Data License Notice:** The datasets provided here are collected from multiple third-party sources. Data licensing and usage terms should be addressed to the individual sources listed below. Please refer to each source's own terms of service and licensing agreements before use.

## Available Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **Humanual-News** | [YouTube](https://www.youtube.com/) | News video comments |
| **Humanual-Book** | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | Book reviews and ratings |
| **Humanual-Opinion** | [Reddit](https://www.reddit.com/) | Subreddit posts and comments |
| **Humanual-Politics** | [Medium](https://medium.com/) | Political blog posts and comments |
| **Humanual-Chat** | [WildChat](https://wildchat.allen.ai/) | Multi-turn AI chat conversations |
| **Humanual-Email** | [Enron Email Corpus](https://www.cs.cmu.edu/~enron/) | Corporate email threads |

## Raw Dataset Format

Each row of the raw dataset follows this schema:

```python
{
    "prompt": [
        # First turn is always the post. Metadata contains post information.
        {"role": poster_id, "content": ..., "metadata": ...},
        # Subsequent turns are user comments. Metadata contains comment information.
        {"role": user_id, "content": ..., "metadata": ...},
        ...
    ],
    "completion": ...,  # str: the target comment
    "post_id": ...,     # str: globally unique post identifier
    "user_id": ...,     # str: globally unique user identifier
    "timestamp": ...,   # int: UTC timestamp
    "metadata": ...,    # metadata for the completion comment
}
```

---

## Step 1: Collecting Raw Data

Each dataset has its own scraper module. Below are instructions for collecting raw data from each source.

### Humanual-News

Scrapes YouTube video metadata, comments, and optionally transcripts.

**Required environment variables:**
- `YOUTUBE_API_KEY`: YouTube Data API key(s). Multiple keys can be separated by `:` for quota rotation.
- `WEBSHARE_PROXY_USERNAME` and `WEBSHARE_PROXY_PASSWORD`: Proxy credentials for transcript fetching (see [Webshare](https://www.webshare.io)).

```shell
# Step 1: Scrape video metadata and comments
python -m humanual_datasets.humanual_news \
    --channels BBCNews CNN \
    --push_to_hub <your-hf-org>/humanual_news_raw_dataset \
    --scrape_raw \
    --config .env

# Step 2 (optional): Fetch video transcripts separately
python -m humanual_datasets.humanual_news \
    --channels BBCNews CNN \
    --push_to_hub <your-hf-org>/humanual_news_raw_dataset \
    --transcripts \
    --config .env
```

| Argument | Description |
|----------|-------------|
| `--channels` | **(required)** YouTube channel usernames, or a JSON file mapping channels to playlist names. |
| `--scrape_raw` | Scrape video metadata and comments from the YouTube API. |
| `--transcripts` | Fetch video transcripts (run separately from `--scrape_raw`). |
| `--top_level_only` | Only fetch top-level comments (exclude replies). |
| `--push_to_hub` | **(required)** HuggingFace Hub repo for the raw dataset. |
| `--config` | Path to `.env` file with API credentials. |
| `--max_videos_per_channel` | Limit videos per channel (for testing). |
| `--verification_mode` | HuggingFace loading verification: `no_checks`, `basic_checks`, `all_checks`. Default: `basic_checks`. |
| `--no_create_raw` | Skip creating the raw dataset (useful for transcript-only runs). |

**Tip:** Run `--scrape_raw` and `--transcripts` as separate steps. Scraping can fail due to API quota exhaustion, and separating the steps makes recovery easier.

The `--channels` argument also accepts a JSON file to specify playlist names per channel:
```json
{
    "BBCNews": ["US & Canada | BBC News", "Sport | BBC News", "Health | BBC News"],
    "CNN": ["World News", "Entertainment", "Science & Technology"]
}
```

### Humanual-Book

Collects product reviews from the Amazon Reviews 2023 dataset via HuggingFace.

**Required environment variables:** None (data is publicly available on HuggingFace).

```shell
python -m humanual_datasets.humanual_book \
    --categories Books \
    --push_to_hub <your-hf-org>/humanual_book_raw_dataset
```

| Argument | Description |
|----------|-------------|
| `--categories` | **(required)** List of Amazon product categories to collect (e.g., `Books`, `All_Beauty`). |
| `--data_dirname` | Local directory for intermediate data. Default: `data`. |
| `--category_splits` | Split each category into N parts to reduce memory. Default: `1`. |
| `--push_to_hub` | HuggingFace Hub repo to upload the raw dataset. |
| `--config` | Path to `.env` file for credentials. |
| `--max_items_per_category` | Limit items per category (for debugging). |

### Humanual-Opinion

Collects posts and comments from Reddit subreddits via the Reddit API (PRAW).

**Required environment variables:**
- `REDDIT_USER_AGENT`: Reddit API user agent string.
- `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`: Reddit app credentials.
  - Alternatively: `REDDIT_USERNAME` and `REDDIT_PASSWORD` for script-type authentication.
  - Supports multiple credential sets: `REDDIT_CLIENT_ID_1`, `REDDIT_CLIENT_ID_2`, etc.

See [PRAW documentation](https://praw.readthedocs.io/en/stable/getting_started/authentication.html) for setup.

```shell
python -m humanual_datasets.humanual_opinion \
    --subreddits AmItheAsshole \
    --push_to_hub <your-hf-org>/humanual_opinion_raw_dataset \
    --config_path .env
```

| Argument | Description |
|----------|-------------|
| `--subreddits` | **(required)** Subreddit names to scrape (without `r/`). |
| `--sort` | Sort method: `hot`, `new`, `top`, `controversial`, `rising`. Default: `top`. |
| `--time_filter` | Time filter: `all`, `hour`, `day`, `week`, `month`, `year`. Default: `all`. |
| `--max_posts_per_subreddit` | Limit posts per subreddit. |
| `--max_comments_per_post` | Limit top-level comments per post. |
| `--max_comments_per_user` | Limit comments per user for post extension. |
| `--push_to_hub` | HuggingFace Hub repo to upload the raw dataset. |
| `--config_path` | Path to `.env` file with Reddit credentials. |
| `--extend` | Extend posts from user comment history. |

### Humanual-Politics

Scrapes political articles and comments from Medium via RapidAPI.

**Required environment variables:**
- `X-RapidAPI-Key`: API key for the Medium API on RapidAPI.

```shell
python -m humanual_datasets.humanual_politics \
    --tag politics \
    --hf_repo <your-hf-org>/humanual_politics_raw_dataset \
    --config .env
```

| Argument | Description |
|----------|-------------|
| `--tag` | **(required)** Medium tag slug to scrape (e.g., `politics`, `health`). |
| `--years_ago` | Target year as `current_year - years_ago`. Default: `1`. |
| `--months` | Months to scrape (e.g., `1-12`, `1,2,3`). Default: `1-12`. |
| `--out_dir` | Local directory for JSONL outputs. Default: `out_medium_rows`. |
| `--hf_repo` | HuggingFace Hub repo to upload the raw dataset. |
| `--flush_every_calls` | Process and push after this many API calls. Default: `2`. |
| `--article_concurrency` | Parallel article processing count. Default: `5`. |
| `--public` | Make the HuggingFace dataset public. |

### Humanual-Chat

Processes multi-turn AI conversations from the WildChat dataset (publicly available on HuggingFace).

**Required environment variables:** None (data is publicly available on HuggingFace).

```shell
python -m humanual_datasets.humanual_chat \
    --push_to_hub <your-hf-org>/humanual_chat_raw_dataset
```

| Argument | Description |
|----------|-------------|
| `--push_to_hub` | HuggingFace Hub repo to upload the raw dataset. |
| `--config` | Path to `.env` file for credentials. |
| `--max_conversations` | Limit conversations to process (for testing). |

### Humanual-Email

Processes Enron email CSV files into threaded conversations.

**Required environment variables:** None (requires a local Enron CSV file).

**Prerequisite:** You need to first parse the raw Enron maildir into CSV using `utils_parser.py`:
```shell
python -m humanual_datasets.utils_parser \
    --input_csv path/to/enron_emails.csv \
    --output_json path/to/enron_nested.json
```

Then create the raw dataset:
```shell
python -m humanual_datasets.humanual_email \
    --csv_path path/to/enron_emails.csv \
    --push_to_hub <your-hf-org>/humanual_email_raw_dataset
```

| Argument | Description |
|----------|-------------|
| `--csv_path` | **(required)** Path to the Enron emails CSV file. |
| `--push_to_hub` | HuggingFace Hub repo to upload the raw dataset. |
| `--config` | Path to `.env` file for credentials. |
| `--min_thread_size` | Minimum emails per thread to include. Default: `2`. |
| `--max_rows` | Only read first N CSV rows (for testing). |
| `--save_to_disk` | Save the dataset locally via `Dataset.save_to_disk()`. |

---

## Step 2: Processing Raw Data

Use `process_raw.py` to filter, generate user personas, partition, and upload the processed dataset. This script works with any of the datasets above.

### Processing Pipeline

```
Load raw dataset (from --pull_from_hub or --cache_dir or --cache_hub)
        |
Filter by --min_num_comments and --max_num_comments
        |
Apply customized_filter_fn (if defined in <dataset_name>.py)
        |
Apply complete_dataset_fn (if defined in <dataset_name>.py)
        |
Generate user personas (using LLM, based on earliest --persona_history_length comments)
        |
Partition into train / val / seen_test / unseen_test
        |
Upload / save the processed dataset
```

### Example Commands

#### Humanual-News
```shell
python -m humanual_datasets.process_raw \
    --config .env \
    --dataset_name humanual_news \
    --splits BBCNews CNN \
    --subset_mode \
    --global_frac 0.25 \
    --pull_from_hub <your-hf-org>/humanual_news_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_news_processed \
    --min_num_comments 25 \
    --max_num_comments 50 \
    --persona_history_length 20 \
    --save_dir data/humanual_news_processed \
    --cache_dir data/humanual_news_raw \
    --partition_by post \
    --llm_model claude-sonnet-4-5-20250929 \
    --val_frac 0.002 \
    --unseen_frac 0.01 \
    --seen_frac 0.01 \
    --memory_friendly \
    --resume_from_cache
```

#### Humanual-Book
```shell
python -m humanual_datasets.process_raw \
    --dataset_name humanual_book \
    --splits Books \
    --pull_from_hub <your-hf-org>/humanual_book_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_book_processed \
    --min_num_comments 100 \
    --max_num_comments 1000 \
    --persona_history_length 20 \
    --subset_mode \
    --memory_friendly \
    --cache_dir data \
    --resume_from_cache
```

#### Humanual-Opinion
```shell
python -m humanual_datasets.process_raw \
    --dataset_name humanual_opinion \
    --splits AmItheAsshole \
    --pull_from_hub <your-hf-org>/humanual_opinion_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_opinion_processed \
    --min_num_comments 10 \
    --max_num_comments 135 \
    --persona_history_length 20 \
    --cache_dir data/humanual_opinion_raw \
    --save_dir data/humanual_opinion_processed \
    --partition_by post \
    --compute_distribution \
    --resume_from_cache \
    --max_concurrent_users 15 \
    --max_concurrent_posts 20 \
    --max_concurrent_comments 3 \
    --post_dist_batch_size 20 \
    --compute_distribution_no_train
```

#### Humanual-Politics
```shell
python -m humanual_datasets.process_raw \
    --config .env \
    --dataset_name humanual_politics \
    --splits politics health love entrepreneurship travel culture self_improvement \
    --pull_from_hub <your-hf-org>/humanual_politics_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_politics_processed \
    --min_num_comments 20 \
    --max_num_comments 100 \
    --persona_history_length 20 \
    --cache_dir data/humanual_politics_raw \
    --save_dir data/humanual_politics_processed \
    --partition_by post \
    --llm_model claude-sonnet-4-5-20250929 \
    --compute_distribution \
    --resume_from_cache \
    --concurrency 5
```

#### Humanual-Chat
```shell
python -m humanual_datasets.process_raw \
    --dataset_name humanual_chat \
    --splits train \
    --pull_from_hub <your-hf-org>/humanual_chat_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_chat_processed \
    --min_total_turns 5 \
    --max_total_turns 10 \
    --min_num_comments 5 \
    --max_num_comments 10 \
    --min_turns_for_train 6 \
    --val_frac 0.05 \
    --unseen_frac 0.05 \
    --seen_frac 0.05 \
    --fixed_persona "A user who is chatting with an AI assistant." \
    --save_dir data/humanual_chat_processed \
    --partition_by turn \
    --persona_history_length 0 \
    --overwrite
```

#### Humanual-Email
```shell
python -m humanual_datasets.process_raw \
    --config .env \
    --dataset_name humanual_email \
    --splits default \
    --pull_from_hub <your-hf-org>/humanual_email_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_email_processed \
    --min_num_comments 10 \
    --max_num_comments 500 \
    --persona_history_length 20 \
    --cache_dir data/humanual_email_raw \
    --save_dir data/humanual_email_processed \
    --partition_by post \
    --llm_model claude-sonnet-4-5-20250929 \
    --resume_from_cache
```

### process_raw.py Argument Reference

#### Core Arguments

| Argument | Description |
|----------|-------------|
| `--dataset_name` | **(required)** Dataset to process: `humanual_news`, `humanual_book`, `humanual_opinion`, `humanual_politics`, `humanual_chat`, `humanual_email`. |
| `--splits` | **(required)** Split names from the raw dataset (supports Python regex, e.g., `".*"` for all). |
| `--config` | Path to `.env` file with API credentials for LLM-based persona generation. |

#### Filtering

| Argument | Default | Description |
|----------|---------|-------------|
| `--min_num_comments` | `10` | Exclude users with fewer comments than this threshold. |
| `--max_num_comments` | `1000` | Exclude users with more comments than this threshold. |
| `--max_samples` | `None` | Limit total samples (for testing/preview). |

#### Persona Generation

| Argument | Default | Description |
|----------|---------|-------------|
| `--persona_history_length` | `20` | Number of a user's earliest comments used to generate their persona. Set to `0` to skip persona generation. |
| `--remove_used_persona_rows` | `False` | Remove the rows used for persona generation from the final dataset. |
| `--fixed_persona` | `None` | Use a fixed persona string for all users (bypasses LLM generation). |
| `--preview_personas` | `False` | Preview which comments would be used for persona generation without making LLM calls. |
| `--max_comment_length` | `3000` | Remove comments longer than this for persona generation. |
| `--truncate_comment_length` | `1024` | Truncate comments to this word count for persona generation. Set to `0` to disable. |

#### LLM Configuration (for persona generation)

| Argument | Default | Description |
|----------|---------|-------------|
| `--llm_model` | `anthropic/claude-sonnet-4-5-20250929` | LLM model for persona generation (via [litellm](https://docs.litellm.ai/)). |
| `--llm_temperature` | `0.0` | LLM sampling temperature. |
| `--llm_max_tokens` | `4096` | Maximum tokens for LLM response. |

#### Dataset Partitioning

| Argument | Default | Description |
|----------|---------|-------------|
| `--partition_by` | `user` | Partitioning strategy: `user` (split by user), `post` (split by post), or `turn` (split by conversation turns). |
| `--unseen_frac` | `0.08` | Fraction of users held out as unseen test set. |
| `--seen_frac` | `0.08` | Fraction of seen users' comments held out as seen test set. |
| `--val_frac` | `0.02` | Fraction of seen users' comments held out as validation set. |

#### Distribution Metrics

| Argument | Default | Description |
|----------|---------|-------------|
| `--compute_distribution` | `False` | Compute post distribution metrics after persona generation. |
| `--compute_distribution_no_train` | `False` | Compute distribution metrics only for val+test splits. |
| `--post_dist_batch_size` | `10` | Number of comments per LLM call for distribution metrics. |

#### Turn-based Arguments (for `--partition_by turn`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--min_turns_for_train` | `0` | Minimum number of turns required for training data. |
| `--min_total_turns` | `0` | Minimum total turns in a prompt conversation. |
| `--max_total_turns` | `0` | Maximum total turns in a prompt conversation. |

#### Download / Upload / Saving

| Argument | Default | Description |
|----------|---------|-------------|
| `--pull_from_hub` | `None` | HuggingFace Hub repo to load the raw dataset from. |
| `--push_to_hub` | `None` | HuggingFace Hub repo to upload the processed dataset to. |
| `--save_dir` | `None` | Local directory to save processed parquet files. |
| `--overwrite` | `False` | Allow overwriting an existing HuggingFace Hub repo. |
| `--subset_mode` | `False` | Store channels/categories as HuggingFace subsets instead of splits. |

#### Memory Optimization

| Argument | Default | Description |
|----------|---------|-------------|
| `--memory_friendly` | `False` | Minimize memory footprint (may slow processing). |
| `--cache_dir` | `None` | Cache intermediate datasets to local disk to reduce memory usage. |
| `--cache_hub` | `None` | Load the filtered dataset directly from this HuggingFace Hub repo. |
| `--resume_from_cache` | `False` | Resume from existing cache if available (useful after OOM). |
| `--compression` | `uncompressed` | Compression for cached parquet files: `uncompressed`, `snappy`, `lz4`, `zstd`. |

#### Concurrency

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_concurrent_users` | `5` | Max concurrent LLM calls for user-level persona generation. |
| `--max_concurrent_posts` | `50` | Max concurrent LLM calls for post-level processing. |
| `--max_concurrent_comments` | `3` | Max concurrent LLM calls for comment-level processing. |

### Tips

- **Regex splits:** Use `--splits ".*"` to process all available splits.
- **No personas:** Set `--persona_history_length 0` to skip persona generation entirely.
- **Outlier users:** Use `--max_num_comments` to exclude users with unusually many comments.
- **OOM recovery:** Combine `--cache_dir`, `--memory_friendly`, and `--resume_from_cache` to handle large datasets.
  - To use `--cache_hub`: first run with `--cache_dir` to save a parquet file locally, then load it with `datasets.Dataset.from_parquet()` and `push_to_hub()` to your cache repo.

### Extending with Custom Functions

Each dataset module (e.g., `humanual_opinion.py`) can define optional functions that `process_raw.py` will automatically load:

- `customized_filter_fn(row: dict) -> bool`: Additional row-level filtering logic.
- `complete_dataset_fn(df: pl.DataFrame, split: str) -> pl.DataFrame`: Transform the dataset after filtering.
