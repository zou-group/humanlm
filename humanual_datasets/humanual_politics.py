import os
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Iterable, Optional, AsyncGenerator
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from datetime import datetime, timezone
import argparse

'''
    Get archived articles by month/year, faster
    Data keys/formatting are the same i.e.
      - For articles, metadata.assets == assets["assets"], metadata.recommended == recommended["recommended_articles"]
      - For responses, metadata.assets == assets["assets"]

    user followers and following pages as well as related and rec articles are commented out due to speed
    Note: append jsonls to huggingface later due to speed
'''


BASE = "https://medium2.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": "",
    "X-RapidAPI-Host": "medium2.p.rapidapi.com",
}
RATE = AsyncLimiter(10, 1)  
TIMEOUT = aiohttp.ClientTimeout(total=10)
CONNECTOR_KW = dict(limit=256, limit_per_host=0, ttl_dns_cache=300)


def load_keys():
    load_dotenv()
    key = os.getenv("X-RapidAPI-Key", "")
    if not key:
        raise SystemExit("Missing X-RapidAPI-Key in environment (.env).")
    HEADERS["X-RapidAPI-Key"] = key

from typing import Awaitable, Callable, Any


_inflight = {}

async def _singleflight(key: str, fn: Callable[[], Awaitable[Any]]):
    task = _inflight.get(key)
    if task is not None:
        return await task

    task = asyncio.create_task(fn())
    _inflight[key] = task
    try:
        return await task
    finally:
        _inflight.pop(key, None)


async def fetch_json(session: aiohttp.ClientSession, path: str, params: Dict = None) -> Dict:
    url = f"{BASE}{path}"
    params = params or {}
    for delay in (0, 3, 8, 15):
        if delay:
            await asyncio.sleep(delay)
        try:
            async with RATE:
                async with session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status in (408, 429) or (500 <= r.status < 600):
                        continue
                    text = await r.text()
                    raise RuntimeError(f"HTTP {r.status} {url}: {text[:300]}")
        except (aiohttp.ClientError, asyncio.TimeoutError):
            continue
    raise RuntimeError(f"Failed after retries: {url}")


def to_epoch_utc(dt_str: str) -> int:
    if not dt_str:
        return 0
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)
            return int(dt.timestamp())
        except Exception:
            return 0

def parse_months_arg(months_arg: str) -> List[int]:
    months: Set[int] = set()
    s = months_arg.strip()
    if "-" in s and "," not in s and " " not in s:
        a, b = s.split("-", 1)
        lo, hi = int(a), int(b)
        for m in range(lo, hi + 1):
            if 1 <= m <= 12:
                months.add(m)
    else:
        for tok in s.replace(",", " ").split():
            try:
                m = int(tok)
                if 1 <= m <= 12:
                    months.add(m)
            except Exception:
                continue
    return sorted(months or [1])


article_info_cache: Dict[str, Dict] = {}
article_content_cache: Dict[str, str] = {}
article_assets_cache: Dict[str, Dict] = {}
article_fans_cache: Dict[str, Dict] = {}
article_responses_cache: Dict[str, Dict] = {}

user_bundle_cache: Dict[str, Dict] = {}


async def get_article_info(session, article_id: str) -> Dict:
    if article_id in article_info_cache:
        return article_info_cache[article_id]

    async def _fetch():
        j = await fetch_json(session, f"/article/{article_id}")
        article_info_cache[article_id] = j
        return j

    return await _singleflight(f"article_info:{article_id}", _fetch)


async def get_article_content(session, article_id: str) -> str:
    if article_id in article_content_cache:
        return article_content_cache[article_id]

    async def _fetch():
        j = await fetch_json(session, f"/article/{article_id}/content")
        content = j.get("content", "")
        article_content_cache[article_id] = content
        return content

    return await _singleflight(f"article_content:{article_id}", _fetch)


async def get_article_assets(session, article_id: str) -> Dict:
    if article_id in article_assets_cache:
        return article_assets_cache[article_id]

    async def _fetch():
        j = await fetch_json(session, f"/article/{article_id}/assets")
        article_assets_cache[article_id] = j
        return j

    return await _singleflight(f"article_assets:{article_id}", _fetch)


async def get_article_fans(session, article_id: str) -> Dict:
    if article_id in article_fans_cache:
        return article_fans_cache[article_id]

    async def _fetch():
        j = await fetch_json(session, f"/article/{article_id}/fans")
        article_fans_cache[article_id] = j
        return j

    return await _singleflight(f"article_fans:{article_id}", _fetch)


async def get_article_responses(session, article_id: str) -> Dict:
    if article_id in article_responses_cache:
        return article_responses_cache[article_id]

    async def _fetch():
        j = await fetch_json(session, f"/article/{article_id}/responses")
        article_responses_cache[article_id] = j
        return j

    return await _singleflight(f"article_responses:{article_id}", _fetch)


async def build_article_bundle(session, article_id: str) -> Dict:
    info, content, assets_j, fans, responses = await asyncio.gather(
        get_article_info(session, article_id),
        get_article_content(session, article_id),
        get_article_assets(session, article_id),
        get_article_fans(session, article_id),
        get_article_responses(session, article_id),
    )
    assets_inner = assets_j.get("assets") if isinstance(assets_j, dict) else None
    return {
        "counts": info,
        "assets": assets_inner,
        "fans": fans,
        "responses": responses,
        "content_text": content,
    }


async def build_response_bundle(session, response_id: str) -> Dict:
    info, content, assets_j, fans, responses = await asyncio.gather(
        get_article_info(session, response_id),
        get_article_content(session, response_id),
        get_article_assets(session, response_id),
        get_article_fans(session, response_id),
        get_article_responses(session, response_id),
    )
    assets_inner = assets_j.get("assets") if isinstance(assets_j, dict) else None
    return {
        "counts": info,
        "assets": assets_inner,
        "fans": fans,
        "responses": responses,
        "content_text": content,
    }


async def get_user_bundle(session, user_id):
    if user_id in user_bundle_cache:
        return user_bundle_cache[user_id]

    async def _build_bundle():
        async def user_info():
            return await fetch_json(session, f"/user/{user_id}")

        async def user_articles():
            return await fetch_json(session, f"/user/{user_id}/articles")

        async def user_publication_following():
            return await fetch_json(session, f"/user/{user_id}/publication_following")

        async def user_interests():
            return await fetch_json(session, f"/user/{user_id}/interests")

        async def user_publications():
            return await fetch_json(session, f"/user/{user_id}/publications")

        (info, articles, pub_following, interests, publications) = await asyncio.gather(
            user_info(),
            user_articles(),
            user_publication_following(),
            user_interests(),
            user_publications(),
        )

        bundle = {
            "user": info,
            "articles": articles,
            "publication_following": pub_following,
            "interests": interests,
            "publications": publications,
        }
        user_bundle_cache[user_id] = bundle
        return bundle

    return await _singleflight(f"user_bundle:{user_id}", _build_bundle)



async def build_rows_for_article(session, article_id):
    article_bundle = await build_article_bundle(session, article_id)
    article_info = article_bundle["counts"]
    article_text = article_bundle["content_text"]
    article_author = article_info.get("author", "")
    article_epoch = to_epoch_utc(article_info.get("published_at", ""))

    base_prompt_item = {
        "role": str(article_author), #poster_id 
        "content": article_text,
        "metadata": {
            "counts": article_bundle["counts"],
            "assets": article_bundle["assets"],
            "fans": article_bundle["fans"],
            "responses": article_bundle["responses"],
        },
    }

    rows: List[Dict] = []
    user_ids: Set[str] = set([str(article_author)])

    async def dfs(current_id: str, prompt_chain: List[Dict]):
        cb = await build_response_bundle(session, current_id)
        cinfo = cb["counts"]
        ctext = cb["content_text"]
        cauthor = str(cinfo.get("author", ""))
        user_ids.add(cauthor)

        row = {
            "prompt": prompt_chain,
            "completion": ctext,
            "post_id": current_id,   #response id
            "user_id": cauthor,       #commenter id
            "timestamp": article_epoch,
            "metadata": {
                "counts": cb["counts"],
                "assets": cb["assets"],      
                "fans": cb["fans"],
                "responses": cb["responses"],
            },
        }
        rows.append(row)

        #recursing for nested responses
        children = cb.get("responses", {}).get("responses", []) or []
        if children:
            new_prompt_item = {
                "role": cauthor,
                "content": ctext,
                "metadata": {
                    "counts": cb["counts"],
                    "assets": cb["assets"],  
                    "fans": cb["fans"],
                    "responses": cb["responses"],
                },
            }
            new_chain = prompt_chain + [new_prompt_item]
            for child_id in children:
                await dfs(child_id, new_chain)

    first_level_resp_ids = article_bundle["responses"].get("responses", []) or []
    for rid in first_level_resp_ids:
        await dfs(rid, [base_prompt_item])

    users_dict= {}
    sem = asyncio.Semaphore(8)

    async def fetch_one_user(uid: str):
        async with sem:
            try:
                users_dict[uid] = await get_user_bundle(session, uid)
            except Exception as e:
                users_dict[uid] = {"error": str(e)}

    await asyncio.gather(*(fetch_one_user(u) for u in user_ids))

    return rows, users_dict


async def iterate_archived_pages(
    session: aiohttp.ClientSession, tag: str, year: int, month: int
) -> AsyncGenerator[List[str], None]:
    """
    Yield the list of article IDs for each /archived_articles page call (up to 20 IDs per yield)
    for a given tag/year/month, following `next` until exhausted.
    """
    params = {"year": year, "month": f"{month:02d}"}
    path = f"/archived_articles/{tag}"
    next_token: Optional[str] = None

    while True:
        call_params = dict(params)
        if next_token:
            call_params["next"] = next_token
        data = await fetch_json(session, path, params=call_params)
        batch = data.get("archived_articles", []) or []
        if batch:
            yield [x for x in batch if x]
        next_token = data.get("next")
        if not next_token:
            break


async def process_id_chunk(
    session: aiohttp.ClientSession,
    ids: List[str],
    out_dir: str,
    hf_repo: Optional[str],
    private: bool,
    tag: str,
    chunk_label: str,
    article_concurrency: int = 6,
):

    if not ids:
        return

    print(f"[{chunk_label}] START chunk with {len(ids)} ids", flush=True) 

    async def has_comments(aid: str) -> Tuple[str, bool]:
        try:
            resp = await get_article_responses(session, aid)
            cnt = resp.get("count")
            if cnt is None:
                cnt = len(resp.get("responses", []) or [])
            return aid, bool(cnt and cnt > 0)
        except Exception:
            return aid, False

    checks = await asyncio.gather(*(has_comments(a) for a in ids))
    to_process = [aid for aid, ok in checks if ok]

    print(f"[{chunk_label}] buffered={len(ids)} -> with_comments={len(to_process)}")

    all_rows: List[Dict] = []
    merged_users: Dict[str, Dict] = {}
    sem = asyncio.Semaphore(max(1, int(article_concurrency)))

    async def handle_one(aid: str) -> Tuple[str, List[Dict], Dict, Optional[str]]:
        async with sem:
            try:
                rows, users = await build_rows_for_article(session, aid)
                return aid, rows, users, None
            except Exception as e:
                return aid, [], {}, str(e)

    results = await asyncio.gather(*(handle_one(a) for a in to_process))
    for idx, (aid, rows, users, err) in enumerate(results, 1):
        if err is None:
            if rows:
                all_rows.extend(rows)
                merged_users.update(users)
            print(f"[{chunk_label}] ({idx}/{len(to_process)}) article {aid}: +{len(rows)} rows")
        else:
            print(f"[{chunk_label}] Error processing article {aid}: {err}")

    if all_rows or merged_users:
        rows_path, users_path = append_save_jsonl(all_rows, merged_users, out_dir, stem=f"archive_{tag}")
        print(f"[{chunk_label}] Appended rows -> {rows_path}")
        print(f"[{chunk_label}] Appended users -> {users_path}")

    if hf_repo and all_rows:
        pass
        #print(f"[{chunk_label}] Appending to Hugging Face dataset '{hf_repo}' split '{tag}'")
        #push_rows_to_hf_append(all_rows, hf_repo, split_name=tag, private=private)
        #print(f"[{chunk_label}] HF push complete.")


async def process_tag_top_year_k_pipeline(  
    tag: str,
    years_ago: int,
    months: List[int],
    out_dir: str,
    hf_repo: Optional[str] = None,
    private: bool = True,
    flush_every_calls: int = 2,   
    article_concurrency: int = 6, 
):
    """
    Walk /archived_articles/{tag}?year=YYYY&month=MM&next=... across requested months.
    After every `flush_every_calls` archive calls (each returns up to 20 IDs), process and save/push.
    """
    load_keys()
    target_year = datetime.now(timezone.utc).year - int(years_ago)

    processed_or_buffered: Set[str] = set() 
    buffer_ids: List[str] = []
    calls_in_batch = 0
    chunk_idx = 0

    async with aiohttp.ClientSession(
            timeout=TIMEOUT,
            connector=aiohttp.TCPConnector(**CONNECTOR_KW) 
        ) as session:
        for m in months:
            print(f"Scanning archive for tag='{tag}' year={target_year} month={m:02d} ...")
            async for page_ids in iterate_archived_pages(session, tag, target_year, m):
                for aid in page_ids:
                    if aid not in processed_or_buffered:
                        processed_or_buffered.add(aid)
                        buffer_ids.append(aid)

                calls_in_batch += 1
                if calls_in_batch >= max(1, int(flush_every_calls)):
                    chunk_idx += 1
                    chunk_label = f"chunk#{chunk_idx} y{target_year}m{m:02d}"
                    await process_id_chunk(
                        session, buffer_ids, out_dir, hf_repo, private, tag, chunk_label,
                        article_concurrency=article_concurrency
                    )

                    buffer_ids = []
                    calls_in_batch = 0


        if buffer_ids:
            chunk_idx += 1
            chunk_label = f"chunk#{chunk_idx} y{target_year}m{months[-1]:02d}"
            await process_id_chunk(
                session, buffer_ids, out_dir, hf_repo, private, tag, chunk_label,
                article_concurrency=article_concurrency
            )

# appending to jsonl
def iter_jsonl(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def append_jsonl(path: Path, items: Iterable[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def append_save_jsonl(rows: List[Dict], users: Dict[str, Dict], out_dir: str, stem: str) -> Tuple[Path, Path]:
    out = Path(out_dir)
    rows_p = out / f"{stem}_rows.jsonl"
    users_p = out / f"{stem}_users.jsonl"


    existing_post_ids: Set[str] = set()
    for obj in iter_jsonl(rows_p) or []:
        pid = obj.get("post_id")
        if pid:
            existing_post_ids.add(pid)

    existing_user_ids: Set[str] = set()
    for obj in iter_jsonl(users_p) or []:
        uid = obj.get("user_id")
        if uid:
            existing_user_ids.add(uid)


    new_rows = [r for r in rows if r.get("post_id") not in existing_post_ids]

    def user_bundle_to_record(uid: str, bundle: Dict) -> Dict:
        bid = None
        try:
            bid = bundle.get("user", {}).get("id")
        except Exception:
            pass
        return {"user_id": uid or bid, "bundle": bundle}

    new_user_items = []
    for uid, bundle in users.items():
        rec = user_bundle_to_record(uid, bundle)
        uid_norm = rec.get("user_id")
        if uid_norm and uid_norm not in existing_user_ids:
            new_user_items.append(rec)

    if new_rows:
        append_jsonl(rows_p, new_rows)
    if new_user_items:
        append_jsonl(users_p, new_user_items)

    return rows_p, users_p


# this hf push takes too long - just add to hf at the very end
def push_rows_to_hf_append(
    rows: List[Dict],
    repo_id: str,
    split_name: str,
    private: bool = True,
    token_env: str = "HUGGINGFACE_TOKEN",
):
    from datasets import Dataset, load_dataset, DownloadConfig
    from huggingface_hub import HfApi

    token = os.getenv(token_env)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    existing = None
    try:
        existing = load_dataset(
            repo_id,
            split=split_name,
            token=token,
            download_config=DownloadConfig(force_download=True),
        )
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "split" in msg and "found" in msg:
            existing = None
        else:
            raise

    if existing is not None and "post_id" in existing.column_names:
        existing_ids = set(existing.unique("post_id"))
        rows = [r for r in rows if r.get("post_id") not in existing_ids]

    if not rows:
        print(f"HF append: nothing new to add for split '{split_name}'.")
        return

    if existing is None:
        Dataset.from_list(rows).push_to_hub(
            repo_id,
            token=token,
            split=split_name,
            commit_message=f"init split {split_name} (+{len(rows)} rows)",
        )
        return

    merged = existing.to_list() + rows

    seen = set()
    merged_dedup = []
    for r in merged:
        pid = r.get("post_id")
        if pid is None or pid not in seen:
            merged_dedup.append(r)
            if pid is not None:
                seen.add(pid)

    Dataset.from_list(merged_dedup).push_to_hub(
        repo_id,
        token=token,
        split=split_name,
        commit_message=f"append {len(rows)} new rows to {split_name} (total {len(merged_dedup)})",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build Medium rows by walking archived months for a tag/year and pushing to HF under the tag split, flushing every X archive calls."
    )
    parser.add_argument("--tag", required=True, help="Tag slug (e.g., politics)")
    parser.add_argument("--years_ago", type=int, default=1, help="Target year as 'now.year - years_ago' (default: 1)")
    parser.add_argument(
        "--months",
        type=str,
        default="1-12",
        help="Months to scrape. Examples: '1,2,3' or '1 2 3' or '1-12' (default: 1-12)"
    )
    parser.add_argument("--out_dir", default="out_medium_rows", help="Directory for JSONL outputs (append-safe).")
    parser.add_argument("--hf_repo", default="snap-stanford/multiturn_medium_raw_dataset", help="Optional HF dataset repo id (e.g., yourname/medium-rows)")
    parser.add_argument("--public", action="store_true", help="If set, make HF dataset public.")
    parser.add_argument(
        "--flush_every_calls",
        type=int,
        default=2,
        help="Flush (process+save+push) after this many /archived_articles calls (default: 2)."
    )
    parser.add_argument(
        "--article_concurrency",
        type=int,
        default=5,
        help="Number of articles to process in parallel within each chunk (default: 6)."
    )
    args = parser.parse_args()

    months = parse_months_arg(args.months)

    asyncio.run(process_tag_top_year_k_pipeline( 
        args.tag,
        args.years_ago,
        months,
        args.out_dir,
        hf_repo=args.hf_repo,
        private=not args.public,
        flush_every_calls=args.flush_every_calls,
        article_concurrency=args.article_concurrency,
    ))

if __name__ == "__main__":
    main()
