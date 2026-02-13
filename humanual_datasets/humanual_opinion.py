"""
Subreddit Dataset Scraper
Scrapes subreddit posts/comments for each user who commented.

python -m examples.datasets.reddit \
    --subreddits AmItheAsshole \
    --push_to_hub snap-stanford/reddit_raw_dataset \
    --max_posts_per_subreddit 100000 \
    --max_comments_per_post 100000 \
    --max_comments_per_user 1 \
    --config_path .env \
    --extend
"""

import argparse
import asyncio
import html
import os
from copy import deepcopy
from datasets import Dataset, load_dataset
from enum import Enum
from typing import Any, Dict, List, Optional

from collections import Counter
from tqdm import tqdm
import asyncpraw
from dotenv import load_dotenv

class SortType(Enum):
    """Enum for sorting options"""

    HOT = "hot"
    NEW = "new"
    TOP = "top"
    CONTROVERSIAL = "controversial"
    RISING = "rising"


class TimeFilter(Enum):
    """Enum for time filter options"""

    ALL = "all"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


# Convert args to enums
SORT_MAP = {
    "hot": SortType.HOT,
    "new": SortType.NEW,
    "top": SortType.TOP,
    "controversial": SortType.CONTROVERSIAL,
    "rising": SortType.RISING,
}
TIME_MAP = {
    "all": TimeFilter.ALL,
    "hour": TimeFilter.HOUR,
    "day": TimeFilter.DAY,
    "week": TimeFilter.WEEK,
    "month": TimeFilter.MONTH,
    "year": TimeFilter.YEAR,
}

DELETED_INDICATORS = ['', '[removed]', '[deleted]', '[deleted by user]']

def clean_dict(d: dict) -> dict:
    """
    Recursively clean a dict to make it JSON/Arrow serializable:
    - Remove known bad keys that contain asyncpraw objects.
    - Replace empty dicts with None (prevents Parquet 'struct with no child field' errors).
    """
    if not isinstance(d, dict):
        return d

    bad_keys = {"_submission", "_reddit", "_replies", "_comments_by_id"}
    cleaned = {}

    for k, v in d.items():
        if k in bad_keys:
            continue

        if isinstance(v, dict):
            nested = clean_dict(v)
            cleaned[k] = nested if nested else None   # <- ensure no empty dicts
        elif isinstance(v, (list, tuple)):
            cleaned[k] = [clean_dict(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v

    return cleaned

    
def clean_text(text: str) -> str:
    """Clean HTML entities + normalize text"""
    if not text:
        return ""

    cleaned = html.unescape(text)
    cleaned = cleaned.replace("```", "")
    lines = cleaned.split("\n")
    result_lines = []
    in_quote = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(">"):
            if not in_quote:
                result_lines.append(stripped)
                in_quote = True
        else:
            if in_quote:
                result_lines.append("[...quoted text...]")
                in_quote = False
            result_lines.append(line)

    return "\n".join(result_lines).strip()


class SubredditScraper:
    """Reddit scraper for subreddit posts and comments"""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        # Build kwargs only for provided fields
        reddit_kwargs = dict(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,  # scraping public data; keeps us safe even with creds
        )
        if username and password:
            reddit_kwargs.update(username=username, password=password)

        self.reddit = asyncpraw.Reddit(**reddit_kwargs)

    async def collect_subreddit_posts(
        self, 
        subreddit_name: str, 
        limit: int = 100, 
        sort: SortType = SortType.TOP, 
        time_filter: TimeFilter = TimeFilter.ALL
    ) -> List:
        """Collect posts from subreddit"""
        posts = []
        subreddit = await self.reddit.subreddit(subreddit_name)

        try:
            if sort == SortType.HOT:
                post_generator = subreddit.hot(limit=limit)
            elif sort == SortType.NEW:
                post_generator = subreddit.new(limit=limit)
            elif sort == SortType.TOP:
                post_generator = subreddit.top(time_filter=time_filter.value, limit=limit)
            elif sort == SortType.CONTROVERSIAL:
                post_generator = subreddit.controversial(time_filter=time_filter.value, limit=limit)
            elif sort == SortType.RISING:
                post_generator = subreddit.rising(limit=limit)
            else:
                post_generator = subreddit.hot(limit=limit)

            async for post in post_generator:
                try:
                    print(f"Get post '{post.title}' with {post.num_comments} comments")

                    post_dict = deepcopy(post.__dict__)
                    post_dict.update({"author": str(post.author) if post.author else "NA"})
                    posts.append(post_dict)
                except Exception as e:
                    print(f"Error processing post {post['id']}: {e}")
                    continue

        except Exception as e:
            print(f"Error collecting posts from r/{subreddit_name}: {e}")

        return posts

    async def collect_all_level_comments(
        self,
        post_id: str,
        max_top_level_comments: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Recursively collect all comments (top-level + nested) for a Reddit post,
        traversing lazily and only exploring branches that are not [deleted].

        Returns a list of nested comment dicts with their replies in "replies".
        """
        comments_nested = []

        try:
            post = await self.reddit.submission(post_id)
            await post.load()

            # Only resolve MoreComments at the top level for now
            await post.comments.replace_more(limit=0)

            async def serialize_comment_tree(comment):
                """Recursively serialize a comment and explore its replies only if body is valid."""
                try:
                    author_str = str(comment.author) if comment.author else "NA"
                    body_str = getattr(comment, "body", "").strip()

                    if len(body_str) < 30:
                        print('body_str', body_str)

                    # Skip subtree if comment body is deleted
                    if body_str.lower() in DELETED_INDICATORS:
                        return None
                    
                    comment_dict = deepcopy(comment.__dict__)
                    comment_dict.update({
                        "author": author_str,
                        "subreddit": str(comment.subreddit),
                    })

                    # Small delay between replies
                    await asyncio.sleep(0.1)

                    await comment.replies.replace_more(limit=0)
                    serialized_replies = []
                    for reply in comment.replies:

                        serialized_reply = await serialize_comment_tree(reply)
                        if serialized_reply:
                            serialized_replies.append(clean_dict(serialized_reply))

                    comment_dict["replies"] = serialized_replies
                    return comment_dict

                except Exception as e:
                    print(f"Error serializing comment {getattr(comment, 'id', '[unknown]')}: {e}")
                    return None

            top_level_comments = post.comments
            for idx, comment in enumerate(top_level_comments):
                if max_top_level_comments is not None and idx >= max_top_level_comments:
                    break
                serialized = await serialize_comment_tree(comment)
                if serialized:
                    comments_nested.append(serialized)

        except Exception as e:
            print(f"Error collecting nested comments for post {post_id}: {e}")

        return comments_nested

    async def close(self):
        await self.reddit.close()


class SubredditUserDataset:
    """Dataset creator that includes user personas"""

    def __init__(
        self, 
        subreddits,
        max_posts_per_subreddit = 65536,
        max_comments_per_post = None,
        max_comments_per_user = 100,
        sort: SortType = SortType.TOP,
        time_filter: TimeFilter = TimeFilter.ALL,
        push_to_hub: str = None,
        config_path: Optional[str] = None,
        extend: bool = True,
    ):
        self.subreddits = subreddits
        self.max_posts_per_subreddit = max_posts_per_subreddit
        self.max_comments_per_post = max_comments_per_post
        self.max_comments_per_user = max_comments_per_user

        self.sort = sort
        self.time_filter = time_filter
        self.push_to_hub = push_to_hub
        self.extend = extend

        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()

        self.credential_sets = self._load_reddit_credential_sets()

        # Build a pool of scrapers (one per credential set)
        self.scrapers = self._build_scraper_pool()
        self._scraper_idx = 0

        self._load_resume_dataset()
    
    def _load_resume_dataset(self):
        if self.extend:
            try:
                self.resume_dataset = load_dataset(f"{self.push_to_hub}_extended")
            except Exception:
                self.resume_dataset = load_dataset(self.push_to_hub)
        else:
            try:
                self.resume_dataset = load_dataset(self.push_to_hub)
            except Exception:
                self.resume_dataset = None

    def _load_reddit_credential_sets(self) -> List[Dict[str, Optional[str]]]:
        """
        Load multiple credential sets: (client_id, client_secret, user_agent, username?, password?).
        Looks for _1, _2, ... suffixed env vars. Falls back to single set if none found.
        """
        creds = []
        i = 1
        while True:
            cid = os.getenv(f"REDDIT_CLIENT_ID_{i}")
            csec = os.getenv(f"REDDIT_CLIENT_SECRET_{i}")
            ua = os.getenv(f"REDDIT_USER_AGENT_{i}")
            uname = os.getenv(f"REDDIT_USERNAME_{i}")
            pword = os.getenv(f"REDDIT_PASSWORD_{i}")

            if not cid and not csec and not ua:
                break  # stop when next index not present

            # require at least client creds + user_agent per set
            if not (cid and csec and ua):
                raise ValueError(f"Credential set #{i} is incomplete (need CLIENT_ID, CLIENT_SECRET, USER_AGENT).")

            creds.append({
                "client_id": cid,
                "client_secret": csec,
                "user_agent": ua,
                "username": uname,
                "password": pword,
            })
            i += 1

        # Fallback to single set if no indexed sets found
        if not creds:
            cid = os.getenv("REDDIT_CLIENT_ID")
            csec = os.getenv("REDDIT_CLIENT_SECRET")
            ua = os.getenv("REDDIT_USER_AGENT")
            uname = os.getenv("REDDIT_USERNAME")
            pword = os.getenv("REDDIT_PASSWORD")
            if not (cid and csec and ua):
                raise ValueError("Missing Reddit API credentials. Provide either indexed *_1 set(s) or the single-set vars.")
            creds.append({
                "client_id": cid,
                "client_secret": csec,
                "user_agent": ua,
                "username": uname,
                "password": pword,
            })

        return creds

    def _build_scraper_pool(self) -> List["SubredditScraper"]:
        scrapers = []
        for c in self.credential_sets:
            scrapers.append(
                SubredditScraper(
                    client_id=c["client_id"],
                    client_secret=c["client_secret"],
                    user_agent=c["user_agent"],
                    username=c.get("username"),
                    password=c.get("password"),
                )
            )
        return scrapers

    def _next_scraper(self) -> "SubredditScraper":
        scraper = self.scrapers[self._scraper_idx]
        self._scraper_idx = (self._scraper_idx + 1) % len(self.scrapers)
        return scraper

    async def create_raw_dataset(self) -> List[Dict[str, Any]]:

        for subreddit_name in self.subreddits:
            if self.resume_dataset:
                if subreddit_name in self.resume_dataset:
                    dataset = self.resume_dataset[subreddit_name].to_dict()
                    existing_post_ids = set(dataset['post_id'])
                    dataset = [{k: v[i] for k, v in dataset.items()} for i in range(len(dataset['post_id']))]
            else:
                dataset = []
                existing_post_ids = set()

            scraper = self._next_scraper()

            # Collect posts first
            posts = await scraper.collect_subreddit_posts(
                subreddit_name, 
                self.max_posts_per_subreddit, 
                self.sort, self.time_filter
            )

            print(f"Processing {len(posts)} posts one at a time...")

            post_cnt = 0
            # Process each post individually
            for post in tqdm(posts):
                post_cnt += 1
                
                if post['id'] in existing_post_ids:
                    print(f"Skipping post {post['id']} already in dataset")
                    continue
                
                # Get comments for this post
                nested_comments = await scraper.collect_all_level_comments(
                    post['id'], 
                    self.max_comments_per_post
                )

                # Process this post's data
                post_entries = self._process_single_post(post, nested_comments)
                
                # Clean before adding to dataset
                for entry in post_entries:
                    entry["metadata"] = clean_dict(entry["metadata"])
                    for p in entry["prompt"]:
                        p["metadata"] = clean_dict(p["metadata"])
                        p["metadata"].pop("replies", None)
                    dataset.append(entry)

                # Small delay between posts
                await asyncio.sleep(0.1)

                if dataset and (post_cnt % 5 == 0 or post_cnt == len(posts)):
                    if self.push_to_hub:
                        self.resume_dataset = Dataset.from_list(dataset)
                        self.resume_dataset.push_to_hub(self.push_to_hub, split=subreddit_name, private=True)
        
        if self.extend:
            await self._process_extended_user_posts()

    async def _process_extended_user_posts(self):
        # Collect all unique user IDs from the existing dataset
        user_ids = [
            uid
            for sub in self.subreddits
            for uid in self.resume_dataset[sub]["user_id"]
        ]
        # Count how many times each user_id appears
        user_counts = Counter(user_ids)
        print(f"Found {len(user_counts)} unique users in the dataset.")

        # Filter user_ids with fewer than max_comments_per_user
        valid_users = {uid for uid, count in user_counts.items() if count < self.max_comments_per_user}
        print(f"Users with < {self.max_comments_per_user} comments: {len(valid_users)}")

        # Fetch post IDs from each user's recent history
        extended_post_ids = set()
        for user_id in tqdm(list(valid_users)[:5], desc="Fetching user histories"):
            try:
                user_engaged_post_ids = await self._get_user_engaged_post_ids(user_id)
                extended_post_ids.update(user_engaged_post_ids)
                print(f"Get user '{str(user_id)}' with {len(user_engaged_post_ids)} post ids")
            except Exception as e:
                print(f"Failed to fetch history for {user_id}: {e}")

        # Identify new post IDs not already in the dataset
        existing_post_ids = {
            pid for split, dset in self.resume_dataset.items() for pid in dset["post_id"]
        }
        additional_post_ids = extended_post_ids - existing_post_ids
        print(f"Found {len(additional_post_ids)} additional posts from user histories.")

        # Collect and process new entries by subreddit
        new_entries_by_subreddit = {}

        post_cnt = 0
        for post_id in tqdm(additional_post_ids, desc="Processing extended posts"):
            post_cnt += 1
            try:
                scraper = self._next_scraper()
                post = await scraper.reddit.submission(post_id)
                await post.load()
                post_dict = deepcopy(post.__dict__)
                post_dict.update({"author": str(post.author) if post.author else "NA"})

                nested_comments = await scraper.collect_all_level_comments(
                    post_id, self.max_comments_per_post
                )

                post_entries = self._process_single_post(post_dict, nested_comments)
                if not post_entries:
                    continue

                subreddit = post_entries[0]["prompt"][0]["metadata"]["subreddit"]

                for entry in post_entries:
                    entry["metadata"] = clean_dict(entry["metadata"])
                    for p in entry["prompt"]:
                        p["metadata"] = clean_dict(p["metadata"])
                        p["metadata"].pop("replies", None)

                new_entries_by_subreddit.setdefault(subreddit, []).extend(post_entries)

            except Exception as e:
                print(f"Failed to process post {post_id}: {e}")
                continue

            if new_entries_by_subreddit and (post_cnt % 5 == 0 or post_cnt == len(additional_post_ids)):
                # Merge with existing data and push to Hugging Face Hub
                for subreddit, new_entries in new_entries_by_subreddit.items():
                    if subreddit in self.resume_dataset:
                        old_raw = self.resume_dataset[subreddit].to_dict()
                        old_entries = [
                            {k: v[i] for k, v in old_raw.items()}
                            for i in range(len(old_raw.get("post_id", [])))
                        ]
                        merged = old_entries + new_entries
                    else:
                        merged = new_entries
                    
                    Dataset.from_list(merged).push_to_hub(
                        f"{self.push_to_hub}_extended", 
                        split=subreddit, private=True
                    )
                    print(f"Pushed extended dataset for r/{subreddit} with {len(merged)} total entries.")


    async def _get_user_engaged_post_ids(self, user_id: str) -> List[str]:
        """Get user's last 5 posts and comments"""
        print(f"Fetching history for: {user_id}")
        
        post_ids = []
        
        try:
            user = await self._next_scraper().reddit.redditor(user_id)
            
            # async for submission in user.submissions.new(limit=1):
            #     post_ids.append(submission.id)
            
            async for comment in user.comments.new(limit=self.max_comments_per_user):
                post_ids.append(comment.link_id.split('_')[-1])
                
        except Exception as e:
            print(f"Error fetching data for {user_id}: {e}")
        
        return post_ids
    

    def _process_single_post(
        self,
        post: Dict[str, Any],
        nested_comments: List,
    ) -> List[Dict[str, Any]]:
        """
        Given a post and its nested comments, produce a list of training examples where:
        - each example includes the entire dialogue history (post + parent comments) in 'prompt'
        - the final comment is treated as the 'completion'
        """

        entries = []

        def clean(content):
            return clean_text(content) if isinstance(content, str) else ""

        def build_prompt_path(ancestor_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Build the prompt from post + ancestor chain"""
            prompt = []

            post.pop("comments", None) # pop CommentForest object
            
            # Post is always the first message in the thread
            post_prompt = {
                "role": str(post.get("author", "poster")),
                "content": post["title"] + ("\n\n" + clean(post["selftext"]) if post.get("is_self") and post.get("selftext", "").strip() else ""),
                "metadata": clean_dict({
                    **post,
                    "subreddit": post['subreddit_name_prefixed'].lstrip("r/"),
                    "thread_id": f"t3_{post['id']}",
                })
            }
            prompt.append(post_prompt)

            # Add all ancestor comments
            for ancestor_comment in ancestor_chain:
                prompt.append({
                    "role": ancestor_comment["author"],
                    "content": clean(ancestor_comment["body"]),
                    "metadata": ancestor_comment
                })

            return prompt

        def walk_comment_tree(
            comment: Dict[str, Any],
            ancestor_chain: List[Dict[str, Any]],
            comment_idx: int
        ):
            """Recursively walk the tree and emit entries"""

            # Build prompt and entry
            prompt = build_prompt_path(ancestor_chain)
            completion = clean(comment["body"])

            entry = {
                "prompt": prompt,
                "completion": completion,
                "post_id": post["id"],
                "user_id": comment["author"],
                "timestamp": comment["created_utc"],
                "metadata": comment,
            }

            entries.append(entry)

            # Recurse on replies, adding current comment to ancestor chain
            for reply in comment.get("replies", []):
                walk_comment_tree(reply, ancestor_chain + [comment], comment_idx + 1)

        # Begin recursive tree traversal
        for idx, top_level_comment in enumerate(nested_comments):
            walk_comment_tree(top_level_comment, ancestor_chain=[], comment_idx=idx)

        return entries
    
    async def close(self):
        # Close all underlying Reddit clients
        for s in getattr(self, "scrapers", []):
            await s.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subreddit Dataset Scraper with User Personas")
    parser.add_argument("--subreddits", nargs="+", type=str, required=True, help="Subreddit name (without r/)")
    parser.add_argument("--max_posts_per_subreddit", type=int, default=None, help="Maximum number of posts to collect (default: 100)")
    parser.add_argument(
        "--max_comments_per_post", type=int, default=None, help="Maximum number of top comments per post (default: None = all comments)"
    )
    parser.add_argument(
        "--max_comments_per_user", type=int, default=None, help="Maximum number of comments per user to extend the post ids"
    )
    parser.add_argument(
        "--sort",
        choices=["hot", "new", "top", "controversial", "rising"],
        default="top",
        help="Sort method for posts (default: top)",
    )
    parser.add_argument(
        "--time_filter",
        choices=["all", "hour", "day", "week", "month", "year"],
        default="all",
        help="Time filter for top posts (default: all)",
    )
    parser.add_argument("--push_to_hub", type=str, default=None, help="Push dataset to Hugging Face Hub (optional)")
    parser.add_argument("--config_path", type=str, default=None, help="Path to .env file with Reddit credentials")
    parser.add_argument("--extend", action="store_true", help="Extend posts from user history")

    args = parser.parse_args()

    args.sort = SORT_MAP[args.sort]
    args.time_filter = TIME_MAP[args.time_filter]

    async def main():
        dataset = SubredditUserDataset(
            subreddits=args.subreddits,
            config_path=args.config_path,
            max_posts_per_subreddit=args.max_posts_per_subreddit,
            max_comments_per_post=args.max_comments_per_post,
            max_comments_per_user=args.max_comments_per_user,
            push_to_hub=args.push_to_hub,
            extend=args.extend,
            sort=args.sort, time_filter=args.time_filter
        )
        try:
            await dataset.create_raw_dataset()
        finally:
            await dataset.close()

    asyncio.run(main())