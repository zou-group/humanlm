import argparse
import asyncio
import json
import multiprocessing
import multiprocessing.pool
import os
import os.path as osp
import time
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Optional

import polars as pl
from datasets import (
    Dataset,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from dotenv import load_dotenv
from tqdm.rich import tqdm

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import HttpError

    from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
    from youtube_transcript_api.proxies import WebshareProxyConfig
except ImportError:
    print("Warning: Please install relevant packages")

from .utils_parser import inplace_clean_empty_dict

VIDEO_FEATURES = Features(
    {
        "kind": Value(dtype="string", id=None),
        "etag": Value(dtype="string", id=None),
        "id": Value(dtype="string", id=None),
        "snippet": {
            "categoryId": Value(dtype="string", id=None),
            "channelId": Value(dtype="string", id=None),
            "channelTitle": Value(dtype="string", id=None),
            "defaultAudioLanguage": Value(dtype="string", id=None),
            "defaultLanguage": Value(dtype="string", id=None),
            "description": Value(dtype="string", id=None),
            "liveBroadcastContent": Value(dtype="string", id=None),
            "localized": {"description": Value(dtype="string", id=None), "title": Value(dtype="string", id=None)},
            "publishedAt": Value(dtype="string", id=None),
            "tags": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "thumbnails": {
                "default": {
                    "height": Value(dtype="int64", id=None),
                    "url": Value(dtype="string", id=None),
                    "width": Value(dtype="int64", id=None),
                },
                "high": {
                    "height": Value(dtype="int64", id=None),
                    "url": Value(dtype="string", id=None),
                    "width": Value(dtype="int64", id=None),
                },
                "maxres": {
                    "height": Value(dtype="int64", id=None),
                    "url": Value(dtype="string", id=None),
                    "width": Value(dtype="int64", id=None),
                },
                "medium": {
                    "height": Value(dtype="int64", id=None),
                    "url": Value(dtype="string", id=None),
                    "width": Value(dtype="int64", id=None),
                },
                "standard": {
                    "height": Value(dtype="int64", id=None),
                    "url": Value(dtype="string", id=None),
                    "width": Value(dtype="int64", id=None),
                },
            },
            "title": Value(dtype="string", id=None),
        },
        "contentDetails": {
            "caption": Value(dtype="string", id=None),
            "contentRating": {"ytRating": Value(dtype="string", id=None)},
            "definition": Value(dtype="string", id=None),
            "dimension": Value(dtype="string", id=None),
            "duration": Value(dtype="string", id=None),
            "licensedContent": Value(dtype="bool", id=None),
            "projection": Value(dtype="string", id=None),
            "regionRestriction": {
                "allowed": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
                "blocked": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            },
        },
        "statistics": {
            "commentCount": Value(dtype="string", id=None),
            "favoriteCount": Value(dtype="string", id=None),
            "likeCount": Value(dtype="string", id=None),
            "viewCount": Value(dtype="string", id=None),
        },
        "video_id": Value(dtype="string", id=None),
        "title": Value(dtype="string", id=None),
        "description": Value(dtype="string", id=None),
    }
)


TRANSCRIPTS_FEATURES = Features(
    {
        "video_id": Value(dtype="large_string", id=None),
        "video_transcript": Value(dtype="large_string", id=None),
    }
)


def datetime_to_utc_timestamp(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    dt = dt.replace(tzinfo=timezone.utc)
    timestamp = dt.timestamp()
    return int(timestamp)


def retry_if_quota_issue(fn, interval=7200):
    def wrapper(*args, **kwargs):
        while True:
            try:
                return fn(*args, **kwargs)
            except HttpError as e:
                if (
                    e.resp.status == 403
                    and e.reason.find(
                        """The request cannot be completed because you have exceeded your <a href="/youtube/v3/getting-started#quota">quota</a>"""
                    )
                    != -1
                ):
                    youtube_dataset = args[0]
                    assert type(youtube_dataset) == YoutubeDataset
                    if youtube_dataset.using_youtube_api_key == youtube_dataset.youtube_api_keys[-1]:
                        if youtube_dataset.current_playlist_info is not None:
                            youtube_dataset.save_playlist_data_to_disk(*youtube_dataset.current_playlist_info)
                        print(f"[{datetime.now()}] All API keys exhausted.")
                        print(
                            f"[{datetime.now()}] Any current states are saved as above. You can kill me and rerun immediately if you get other keys; otherwise I will sleep for {interval} seconds and retry..."
                        )
                        time.sleep(interval)
                    else:
                        print(
                            f"[{datetime.now()}] API key {youtube_dataset.using_youtube_api_key} exhausted, switching to next key..."
                        )
                    youtube_dataset.reset_connection()
                else:
                    raise e

    return wrapper


class YoutubeDataset:
    def __init__(
        self,
        channels: dict[str, list[str]],
        max_videos_per_channel: Optional[int] = None,
        transcripts: bool = False,
        top_level_only: bool = False,
        push_to_hub: str = None,
        verification_mode: str = "no_checks",
        config_path: Optional[str] = None,
    ):
        self.max_videos_per_channel = max_videos_per_channel
        self.channels: dict[str, list[str]] = channels
        self.transcripts = transcripts
        self.top_level_only = top_level_only
        self.config_path = config_path
        self.verification_mode = verification_mode
        if config_path is not None:
            load_dotenv(config_path)
        else:
            load_dotenv()
        self.youtube_api_keys = os.getenv("YOUTUBE_API_KEY", None)
        if self.youtube_api_keys is None:
            raise ValueError("YOUTUBE_API_KEY is not set in the environment variables.")
        self.youtube_api_keys = self.youtube_api_keys.split(":")
        self.youtube_api_key_gen = self.next_api_key_generator()
        self.proxy_username = os.getenv("WEBSHARE_PROXY_USERNAME", None)
        self.proxy_password = os.getenv("WEBSHARE_PROXY_PASSWORD", None)
        if not all([self.proxy_username, self.proxy_password]):
            raise ValueError(
                "WEBSHARE_PROXY_USERNAME or WEBSHARE_PROXY_PASSWORD is not set in the environment variables. This is essential to avoid being blocked by Youtube."
            )
        self.cache_dir = os.getenv("YOUTUBE_CACHE_DIR", None)
        if self.cache_dir is None:
            raise ValueError("YOUTUBE_CACHE_DIR is not set in the environment variables.")
        self.cache_dir: Path = Path(self.cache_dir)
        self.youtube = None
        self.ytt_api = None
        self.reset_connection()
        self.push_to_hub = push_to_hub

        self.current_playlist_info = None  # For retrier to save state.

    def next_api_key_generator(self):
        while True:
            for api_key in self.youtube_api_keys:
                yield api_key

    def reset_connection(self):
        self.using_youtube_api_key = next(self.youtube_api_key_gen)
        self.youtube = build("youtube", "v3", developerKey=self.using_youtube_api_key)
        self.ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=self.proxy_username,
                proxy_password=self.proxy_password,
            )
        )

    @retry_if_quota_issue
    def get_channel(self, channel_name: str):
        response = self.youtube.channels().list(part="snippet", forUsername=channel_name).execute()
        if "items" not in response or len(response["items"]) == 0:
            raise ValueError(f"Channel with name {channel_name} not found.")
        return response["items"][0]

    @retry_if_quota_issue
    def get_playlists(self, channel_id):
        playlists = []
        request = self.youtube.playlists().list(part="snippet", channelId=channel_id, maxResults=50)
        while request:
            response = request.execute()
            playlists.extend(response.get("items", []))
            request = self.youtube.playlists().list_next(request, response)
        return playlists

    @retry_if_quota_issue
    def get_playlist_items(self, playlist_id):
        playlist_items = []
        request = self.youtube.playlistItems().list(part="snippet,contentDetails", playlistId=playlist_id, maxResults=50)
        while request:
            response = request.execute()
            playlist_items.extend(response.get("items", []))
            request = self.youtube.playlistItems().list_next(request, response)
        return playlist_items

    @retry_if_quota_issue
    def get_video(self, video_id):
        response = self.youtube.videos().list(part="snippet,contentDetails,statistics", id=video_id).execute()
        return response["items"][0] if response["items"] else None

    @retry_if_quota_issue
    def get_comments(self, video_id):
        comments = []
        parts = "snippet" if self.top_level_only else "snippet,replies"
        request = self.youtube.commentThreads().list(part=parts, videoId=video_id, maxResults=100)

        while request:
            response = request.execute()
            for item in response.get("items", []):
                top_comment = item["snippet"]["topLevelComment"]
                if self.top_level_only and top_comment["snippet"].get("parentId"):
                    continue
                top_comment["upper_comments"] = []
                comments.append(top_comment)
                if not self.top_level_only:
                    for reply in item.get("replies", {}).get("comments", []):
                        reply["upper_comments"] = [top_comment]
                        comments.append(reply)
            request = self.youtube.commentThreads().list_next(request, response)
        return comments

    def get_transcript(self, video, language_code=None):
        """
        generated_if_available: if generated available, use generated transcript

        Returns:
            The transcript snippets if available;
            None otherwise.
        """
        if language_code is not None:
            # Listing all languages takes too long, so if language_code is given, just use it.
            fetched = self.ytt_api.fetch(video["id"], languages=[language_code])
        else:
            video_id = video["id"]
            language_code = video["snippet"]["defaultLanguage"]
            trans_list = self.ytt_api.list(video_id)
            try:
                used_trans = trans_list.find_generated_transcript(language_code)
            except NoTranscriptFound:
                used_trans = None
                first_manual = None
                for trans in trans_list:
                    if trans.is_generated:
                        used_trans = trans
                        break
                    else:
                        first_manual = trans
                if used_trans is None:
                    used_trans = first_manual
            fetched = used_trans.fetch()
        return " ".join(t.text for t in fetched)

    def get_transcript_with_retry(self, video, language_code=None, retry=2):
        for _ in range(retry):
            try:
                return self.get_transcript(video, language_code=language_code)
            except Exception as e:
                if str(e).lower().find("brok"):
                    pass
        return "Not available due to API call issue."

    def save_playlist_data_to_disk(self, videos, comments, videos_dir, comments_dir, playlist):
        videos = Dataset.from_list(videos, features=VIDEO_FEATURES)
        comments = Dataset.from_list(comments)
        # Save for each playlist to avoid data loss.
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(comments_dir, exist_ok=True)
        json.dump(playlist, open(videos_dir / "info.json", "w"))
        videos.save_to_disk(videos_dir)
        comments.save_to_disk(comments_dir)
        playlist_title = playlist["snippet"]["title"]
        playlist_id = playlist["id"]
        print(f"Locally cached playlist {playlist_title} ({playlist_id}) to {self.cache_dir}")

    async def scrape_raw_data(self):
        # Get existing splits info so that we can do increamental scraping.

        for channel_name, playlist_names in self.channels.items():
            all_videos = []
            all_comments = []
            self.current_channel = channel_name
            self.current_all_videos = all_videos
            self.current_all_comments = all_comments
            channel = self.get_channel(channel_name)
            playlists = self.get_playlists(channel["id"])
            video_ids = set()
            pbar = tqdm(playlists, desc=f"""Channel "{channel_name}" """)

            for playlist in pbar:
                playlist_videos = []
                playlist_comments = []
                playlist_id = playlist["id"]
                pbar.set_postfix(
                    {
                        "#Videos": len(all_videos),
                        "#Comments": len(all_comments),
                        "playlist": playlist["snippet"]["title"],
                        "playlist_id": playlist_id,
                    }
                )
                playlist_items = self.get_playlist_items(playlist_id)
                playlist_title = playlist["snippet"]["title"]
                if playlist_names is not None and playlist_title not in playlist_names:
                    continue
                # Check if the playlist already exists and the videos are complete; skip this playlist if so.
                playlist_videos_dir = self.cache_dir / "videos" / channel_name / playlist_id
                playlist_comments_dir = self.cache_dir / "comments" / channel_name / playlist_id
                # Skip this playlist if it is complete.
                if playlist_videos_dir.exists():
                    try:
                        existing_playlist_videos = Dataset.load_from_disk(playlist_videos_dir)
                        existing_playlist_comments = Dataset.load_from_disk(playlist_comments_dir)
                    except Exception as e:
                        print(f"Failed to load existing playlist {playlist_title} ({playlist_id}): from {playlist_videos_dir}")
                        raise e
                    playlist_videos.extend(existing_playlist_videos.to_list())
                    playlist_comments.extend(existing_playlist_comments.to_list())
                    if len(existing_playlist_videos) == len(playlist_items):
                        print(f"Skip playlist {playlist_title} ({playlist_id}) since already exists.")
                        all_videos.extend(playlist_videos)
                        all_comments.extend(playlist_comments)
                        continue
                    else:
                        print(
                            f"Playlist {playlist_title} ({playlist_id}) exists but incomplete, re-scraping. Existing: {len(existing_playlist_videos)}, Expected: {len(playlist_items)}"
                        )
                        video_ids.update(existing_playlist_videos["video_id"])
                self.current_playlist_info = (
                    playlist_videos,
                    playlist_comments,
                    playlist_videos_dir,
                    playlist_comments_dir,
                    playlist,
                )
                # Otherwise, we re-scrape the entire playlist.
                added_new_to_playlist = False
                v_pbar = tqdm(playlist_items, desc=f"""Playlist "{playlist_title}" """, leave=False)
                for item in v_pbar:
                    # Get video details and process
                    video_id = item["contentDetails"]["videoId"]
                    if video_id in video_ids:
                        continue
                    video_ids.add(video_id)
                    video = self.get_video(video_id)
                    if video is None:
                        # NOTE: Not sure why, but sometimes it doesn't return a corresponding video.
                        continue
                    video["video_id"] = video["id"]
                    video["title"] = video["snippet"]["title"]
                    video["description"] = video["snippet"]["description"]
                    inplace_clean_empty_dict(video)
                    try:
                        comments = self.get_comments(video_id)
                    except HttpError as e:
                        if e.resp.status == 403 and e.resp.reason.find("disabled comments") != -1:
                            # This happens when comments are disabled for the video.
                            comments = []
                    if len(comments) == 0:
                        continue

                    for comment in comments:
                        comment["user_id"] = comment["snippet"]["authorChannelId"]["value"]
                        comment["timestamp"] = datetime_to_utc_timestamp(comment["snippet"]["publishedAt"])
                        comment["video_id"] = comment["snippet"]["videoId"]
                        comment["text"] = comment["snippet"]["textOriginal"]
                        inplace_clean_empty_dict(comment)
                    playlist_videos.append(video)
                    playlist_comments.extend(comments)
                    added_new_to_playlist = True
                    if (
                        self.max_videos_per_channel is not None
                        and len(all_videos) + len(playlist_videos) >= self.max_videos_per_channel
                    ):
                        break
                if added_new_to_playlist:
                    self.save_playlist_data_to_disk(
                        playlist_videos, playlist_comments, playlist_videos_dir, playlist_comments_dir, playlist
                    )
                all_videos.extend(playlist_videos)
                all_comments.extend(playlist_comments)

                v_pbar.close()
                if (
                    self.max_videos_per_channel is not None
                    and len(all_videos) + len(playlist_videos) >= self.max_videos_per_channel
                ):
                    break
            pbar.close()

            all_videos = Dataset.from_list(all_videos, features=VIDEO_FEATURES)
            all_comments = Dataset.from_list(all_comments)
            if self.push_to_hub:
                all_videos.push_to_hub(self.push_to_hub, config_name=f"videos_{channel_name}", split="split0", private=True)
                all_comments.push_to_hub(self.push_to_hub, config_name=f"comments_{channel_name}", split="split0", private=True)

    async def scrape_transcripts(self, concurrency=8, channels=None):
        """
        channels: if None, use self.channels
        """

        def parse_transcript_cache(cache_path: str) -> Dataset:
            data = []
            if not osp.exists(cache_path):
                return None
            with open(cache_path, "r") as f:
                lines = f.readlines()
                assert len(lines) % 3 == 0, "Cache file should be in format of video_id\\ntranscript\\n--------\\n"
                for i in range(0, len(lines), 3):
                    video_id = lines[i].strip()
                    transcript = lines[i + 1].strip()
                    data.append({"video_id": video_id, "video_transcript": transcript})
            if len(data) == 0:
                return None
            return Dataset.from_list(data, features=TRANSCRIPTS_FEATURES)

        ytt_pool = multiprocessing.pool.ThreadPool(concurrency)
        channels = channels if channels is not None else self.channels
        for channel_name in channels:
            existing_video_ids = set()

            # NOTE: Generating transcripts is slow and expensive, we should reuse as much as possible.
            # 1. Try to reuse from local cache.
            cache_path = self.cache_dir / "transcripts" / f"{channel_name}.cache"
            os.makedirs(osp.dirname(cache_path), exist_ok=True)
            cache = open(cache_path, "a+")
            cached_transcripts = parse_transcript_cache(cache_path)
            if cached_transcripts is not None:
                existing_video_ids.update(cached_transcripts.to_polars().select("video_id").unique().to_series().to_list())
            print(f"Loaded {len(existing_video_ids)} cached transcripts from {cache_path}")
            # 2. Try to reuse from hub.
            hub_transcripts = load_dataset(self.push_to_hub, name="video_transcripts").get(channel_name, None)
            if hub_transcripts is not None:
                existing_video_ids.update(hub_transcripts.to_polars().select("video_id").unique().to_series().to_list())
                print(f"Loaded {len(hub_transcripts)} transcripts from hub {self.push_to_hub}")
            # Merge existing transcripts.
            if cached_transcripts is not None and hub_transcripts is not None:
                existing_transcripts = (
                    concatenate_datasets([cached_transcripts, hub_transcripts]).to_polars().group_by("video_id").last()
                )
            elif cached_transcripts is not None:
                existing_transcripts = cached_transcripts.to_polars()
            elif hub_transcripts is not None:
                existing_transcripts = hub_transcripts.to_polars()
            else:
                existing_transcripts = None
            if existing_transcripts is not None:
                existing_transcripts = existing_transcripts.filter(pl.col("video_id") != "Not available due to API call issue.")
                existing_transcripts = Dataset.from_polars(existing_transcripts, features=TRANSCRIPTS_FEATURES)
            print(f"Total existing transcripts: {len(existing_video_ids)}")

            # Start generating missing transcripts.
            all_videos = load_dataset(self.push_to_hub, name=f"videos_{channel_name}").to_polars()
            print(f"Channel {channel_name}: {len(all_videos)} videos")
            all_transcript_tasks = []
            video_transcripts = []
            for video in all_videos.iter_rows(named=True):
                video_id = video["video_id"]
                if video_id in existing_video_ids:
                    continue
                transcript_task = ytt_pool.apply_async(
                    self.get_transcript_with_retry,
                    (video,),
                    {"language_code": video["snippet"]["defaultLanguage"], "retry": 2},
                )
                all_transcript_tasks.append((video_id, transcript_task))
            print(f"[{datetime.now()}] Waiting generation jobs to be done. Every finished job will be saved to disk as cache.")
            for video_id, transcript_task in tqdm(
                all_transcript_tasks, desc=f"Generating transcripts for channel {channel_name}"
            ):
                transcript: str = transcript_task.get()
                transcript = transcript.replace("\n", " ")
                video_transcripts.append({"video_id": video_id, "video_transcript": transcript})
                cache.write(f"{video_id}\n")
                cache.write(transcript)
                cache.write("\n--------\n")
                cache.flush()
            cache.close()
            video_transcripts = Dataset.from_list(video_transcripts)
            print(
                f"Regenerated transcripts for {channel_name}, existing: {len(existing_transcripts) if existing_transcripts else 0}, new: {len(video_transcripts)}"
            )
            if existing_transcripts is not None:
                video_transcripts = concatenate_datasets([existing_transcripts, video_transcripts]).sort("video_id")
            if self.push_to_hub:
                video_transcripts.push_to_hub(
                    self.push_to_hub, config_name=f"video_transcripts_{channel_name}", split="split0", private=True
                )
        ytt_pool.close()
        ytt_pool.join()

    async def create_raw_dataset(self):
        if self.push_to_hub is None:
            raise ValueError("push_to_hub must be specified to pull scraped raw data.")
        config_names = get_dataset_config_names("snap-stanford/youtube_raw_dataset")
        for channel_name in self.channels:
            all_videos = load_dataset(
                self.push_to_hub, name=f"videos_{channel_name}", split="split0", verification_mode=self.verification_mode
            ).to_polars()
            all_comments = load_dataset(
                self.push_to_hub, name=f"comments_{channel_name}", split="split0", verification_mode=self.verification_mode
            ).to_polars()
            if self.transcripts:
                video_ids_from_videos = all_videos.select("video_id").unique().sort("video_id").to_series().to_list()
                transcripts_config_name = f"video_transcripts_{channel_name}"
                if transcripts_config_name in config_names:
                    existing_transcripts = load_dataset(
                        self.push_to_hub,
                        name=transcripts_config_name,
                        split="split0",
                        verification_mode=self.verification_mode,
                    ).to_polars()
                    # Check if we have complete transcripts, if not, we generate them.
                    video_ids_from_transcripts = (
                        existing_transcripts.select("video_id").unique().sort("video_id").to_series().to_list()
                    )
                else:
                    video_ids_from_transcripts = []
                if video_ids_from_videos != video_ids_from_transcripts:
                    print(
                        f"Missing transcripts for videos in {channel_name}, we only have {len(video_ids_from_transcripts)}/{len(video_ids_from_videos)}"
                    )
                    # Generate missing transcripts
                    await self.scrape_transcripts(channels=[channel_name])
                # Reload re-generated complete transcripts
                all_transcripts = load_dataset(self.push_to_hub, name=transcripts_config_name).to_polars()
                video_id_to_transcript = {
                    row["video_id"]: row["video_transcript"] for row in all_transcripts.iter_rows(named=True)
                }
            else:
                video_id_to_transcript = {}
            print(f"Channel {channel_name}: {len(all_videos)} videos, {len(all_comments)} comments")
            video_id_to_videos: dict[str, dict] = {row["video_id"]: row for row in all_videos.iter_rows(named=True)}
            all_entries = []
            for comment in tqdm(all_comments.iter_rows(named=True), desc=f"Creating raw dataset for {channel_name}"):
                video_id = comment["video_id"]
                assert video_id in video_id_to_videos, f"Video ID {video_id} not found for comment {comment['id']}"
                video = video_id_to_videos[video_id]
                video["transcript"] = video_id_to_transcript.get(video_id, "Not available.")
                post_prompt = {
                    "role": video["title"],
                    "content": video["description"],
                    "metadata": json.dumps(video),
                }
                comment_prompts = [
                    {
                        "role": upper_comment["snippet"]["authorChannelId"]["value"],
                        "content": upper_comment["snippet"]["textOriginal"],
                        "metadata": json.dumps(upper_comment),
                    }
                    for upper_comment in chain(comment["upper_comments"])
                ]
                entry = {
                    "prompt": [post_prompt, *comment_prompts],
                    "completion": comment["text"],  # str
                    "post_id": video_id,  # str: a globally unique id for this post.
                    "user_id": comment["user_id"],  # str: a globally unique id for this comment (i.e., the complettion)
                    "timestamp": comment["timestamp"],  # int: UTC timestamp
                    "metadata": json.dumps(comment),  # metadata for this comment (i.e., the completion).
                }
                all_entries.append(entry)
            all_entries = Dataset.from_list(all_entries)
            if self.push_to_hub:
                print(f"[{datetime.now()}] Pushing dataset: {all_entries}")
                all_entries.push_to_hub(self.push_to_hub, config_name=channel_name, split="split0", private=True)


def complete_dataset(ds: pl.DataFrame, raw_hub: str) -> pl.DataFrame:
    """
    Populate the metadata into dataset `ds` based on post_id and user_id.
    To efficiently join, you must add a `split` column to `ds` first.
    """
    config_names = get_dataset_config_names(raw_hub)

    entries = []
    by_config_names = ds.partition_by(("__config_name__", "__split_name__"), as_dict=True)
    for (config_name, split_name), rows in tqdm(by_config_names.items(), desc="Complete metadata", total=len(by_config_names)):
        for prefix in ["videos_", "comments_", "video_transcripts_"]:
            if f"{prefix}{config_name}" not in config_names:
                raise ValueError(
                    f"{prefix.capitalize()} subset for {config_name} not found in {raw_hub}. Available configs: {config_names}"
                )
        transcripts = load_dataset(raw_hub, name=f"video_transcripts_{config_name}", split=split_name).to_polars()
        transcripts = {row["video_id"]: row for row in transcripts.iter_rows(named=True)}
        for row in tqdm(rows.iter_rows(named=True), desc=f"Complete metadata for {config_name}", leave=False, total=len(rows)):
            post_id = row["post_id"]
            video_meta = json.loads(row["prompt"][0]["metadata"])
            transcript = transcripts[post_id]["video_transcript"]
            # Truncate transcript to 2048 words
            words = transcript.split()
            if len(words) > 2048:
                transcript = " ".join(words[:2048])
            video_meta["transcript"] = transcript
            row["prompt"][0]["metadata"] = json.dumps(video_meta)
            entries.append(row)
    print(f"[{datetime.now()}] Completed metadata. Now converting back to entries...")
    return pl.DataFrame(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Youtube Raw Dataset.")
    parser.add_argument(
        "--channels", nargs="+", type=str, required=True, help="List of YouTube channel usernames to fetch data from."
    )
    parser.add_argument(
        "--max_videos_per_channel",
        type=int,
        default=None,
        help="For testing purpose, limit the number of videos to fetch per channel",
    )
    parser.add_argument(
        "--scrape_raw",
        action="store_true",
        help="Scrape raw data from YouTube API (you should skip if already scraped)",
    )
    parser.add_argument(
        "--transcripts",
        action="store_true",
        help="Whether to generate video transcripts",
    )
    parser.add_argument(
        "--no_create_raw",
        action="store_true",
        help="Whether to skip creating raw dataset",
    )
    parser.add_argument(
        "--top_level_only",
        action="store_true",
        help="Only fetch top-level comments, excluding replies",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        required=True,
        help="The raw dataset name in Hugging Face Hub",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to .env file with necessary credentials (e.g., OpenAI API key)",
    )
    parser.add_argument(
        "--verification_mode",
        type=str,
        default="basic_checks",
        choices=["no_checks", "basic_checks", "all_checks"],
        help="Loading data from hub might fail due to some stupid issues; you can try to set this to 'no_checks' to skip all verification.",
    )
    args = parser.parse_args()

    if len(args.channels) == 0:
        raise ValueError("At least one channel must be specified.")
    elif len(args.channels) == 1:
        channel_name = args.channels[0]
        if channel_name.endswith(".json"):
            args.channels = json.load(open(channel_name))
        else:
            args.channels = {channel_name: None for channel_name in args.channels}
    else:
        args.channels = {channel_name: None for channel_name in args.channels}
    # After processing, args.channels: dict[str, list[str]]

    dataset = YoutubeDataset(
        channels=args.channels,
        max_videos_per_channel=args.max_videos_per_channel,
        transcripts=args.transcripts,
        top_level_only=args.top_level_only,
        push_to_hub=args.push_to_hub,
        verification_mode=args.verification_mode,
        config_path=args.config,
    )

    if args.scrape_raw:
        asyncio.run(dataset.scrape_raw_data())

    if args.transcripts:
        asyncio.run(dataset.scrape_transcripts())

    if not args.no_create_raw:
        asyncio.run(dataset.create_raw_dataset())
