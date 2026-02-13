"""
Original dataset: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from datasets import Dataset, Features, Sequence, Value, load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from .utils_parser import BYELLOW, RST, memory_usage

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

CATEGORIES = [
    "Subscription_Boxes",  # 16200
    "Magazine_Subscriptions",  # 71500
    "Digital_Music",  # 130400
    "Gift_Cards",  # 152400
    "Health_and_Personal_Care",  # 494100
    "Handmade_Products",  # 664200
    "All_Beauty",  # 701500
    "Appliances",  # 2100000
    "Amazon_Fashion",  # 2500000
    "Musical_Instruments",  # 3000000
    "Video_Games",  # 4600000
    "CDs_and_Vinyl",  # 4800000
    "Software",  # 4900000
    "Industrial_and_Scientific",  # 5200000
    "Baby_Products",  # 6000000
    "Arts_Crafts_and_Sewing",  # 9000000
    "Office_Products",  # 12800000
    "Grocery_and_Gourmet_Food",  # 14300000
    "Toys_and_Games",  # 16300000
    "Patio_Lawn_and_Garden",  # 16500000
    "Pet_Supplies",  # 16800000
    "Movies_and_TV",  # 17300000
    "Sports_and_Outdoors",  # 19600000
    "Automotive",  # 20000000
    "Cell_Phones_and_Accessories",  # 20800000
    "Beauty_and_Personal_Care",  # 23900000
    "Health_and_Household",  # 25600000
    "Kindle_Store",  # 25600000
    "Tools_and_Home_Improvement",  # 27000000
    "Books",  # 29500000
    "Electronics",  # 43900000
    "Unknown",  # 63800000
    "Clothing_Shoes_and_Jewelry",  # 66000000
    "Home_and_Kitchen",  # 6740000
]


@dataclass
class ItemMeta:
    # Main category (i.e., domain) of the product.
    main_category: str
    # Name of the product.
    title: str
    # Rating of the product shown on the product page.
    average_rating: float
    # Number of ratings in the product.
    rating_number: int
    # Bullet-point format features of the product.
    features: list[str]
    # Description of the product.
    description: list[str]
    # Price in US dollars (at time of crawling).
    price: float
    # Images of the product. Each image has different sizes (thumb, large, hi_res). The “variant” field shows the position of image.
    images: list[dict]
    # Videos of the product including title and url.
    videos: list[dict]
    # Store name of the product.
    store: str
    # Hierarchical categories of the product.
    categories: list[str]
    # Product details, including materials, brand, sizes, etc.
    details: dict
    # Parent ID of the product.
    parent_asin: str
    # Recommended bundles from the websites.
    bought_together: list[str]


@dataclass
class Review:
    # Rating of the product (from 1.0 to 5.0).
    rating: float
    # Title of the user review.
    title: str
    # Text body of the user review.
    text: str
    # Images that users post after they have received the product. Each image has different sizes (small, medium, large), represented by the small_image_url, medium_image_url, and large_image_url respectively.
    images: list[dict]
    # ID of the product.
    asin: str
    # Parent ID of the product. Note: Products with different colors, styles, sizes usually belong to the same parent ID. The “asin” in previous Amazon datasets is actually parent ID. Please use parent ID to find product meta.
    parent_asin: str
    # ID of the reviewer
    user_id: str
    # Time of the review (unix time)
    timestamp: int
    # User purchase verification
    verified_purchase: bool
    # Helpful votes of the review
    helpful_vote: int


REVIEW_FEATURES = Features(
    {
        "rating": Value(dtype="float64", id=None),
        "title": Value(dtype="string", id=None),
        "text": Value(dtype="string", id=None),
        "images": [
            {
                "attachment_type": Value(dtype="string", id=None),
                "large_image_url": Value(dtype="string", id=None),
                "medium_image_url": Value(dtype="string", id=None),
                "small_image_url": Value(dtype="string", id=None),
            }
        ],
        "asin": Value(dtype="string", id=None),
        "parent_asin": Value(dtype="string", id=None),
        "user_id": Value(dtype="string", id=None),
        "timestamp": Value(dtype="int64", id=None),
        "verified_purchase": Value(dtype="bool", id=None),
        "helpful_vote": Value(dtype="int64", id=None),
    }
)

VIDEO_FEATURES = Features(
    {
        "main_category": Value(dtype="string", id=None),
        "title": Value(dtype="string", id=None),
        "average_rating": Value(dtype="float64", id=None),
        "rating_number": Value(dtype="int64", id=None),
        "features": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        "description": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        "price": Value(dtype="string", id=None),  # NOTE: This has to be string, because it can be None.
        "images": {
            "hi_res": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "large": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "thumb": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "variant": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        },
        "videos": {
            "title": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "url": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "user_id": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        },
        "store": Value(dtype="string", id=None),
        "categories": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        "details": Value(dtype="string", id=None),
        "parent_asin": Value(dtype="string", id=None),
        "bought_together": Sequence(Value(dtype="string", id=None), length=-1, id=None),
    }
)

# NOTE: I still keep formats above in case we want it in the future.
# But for now, it looks like string is the best solution.
VIDEO_FEATURES = Value(dtype="string", id=None)
REVIEW_FEATURES = Value(dtype="string", id=None)
FEATURES = Features(
    {
        "prompt": [
            {
                "content": Value(dtype="string", id=None),
                "metadata": VIDEO_FEATURES,
                "role": Value(dtype="string", id=None),
            }
        ],
        "completion": Value(dtype="string", id=None),
        "post_id": Value(dtype="string", id=None),
        "user_id": Value(dtype="string", id=None),
        "timestamp": Value(dtype="int64", id=None),
        "metadata": REVIEW_FEATURES,
    }
)


class AmazonReviewDataset:
    def __init__(
        self,
        categories: list[str],
        category_splits: int = 1,
        push_to_hub: str = None,
        config_path: Optional[str] = None,
        *,
        max_items_per_category: Optional[int] = None,
    ):
        self.config_path = config_path
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()
        self.categories = categories
        self.category_splits = category_splits
        self.push_to_hub = push_to_hub

        self.max_items_per_category = max_items_per_category

    async def create_raw_dataset(self):
        """
        NOTE: To save memory, we won't collect all datasets of different categories and return them.
        """
        for category in self.categories:
            if category == "Unknown":
                print("Warning: we deliberately skip the Unknown category, since data may be incomplete.")
                continue
            all_metas = load_dataset(DATASET_NAME, f"raw_meta_{category}", trust_remote_code=True)["full"].to_polars()
            all_reviews = load_dataset(DATASET_NAME, f"raw_review_{category}", trust_remote_code=True)["full"].to_polars()
            print(f"Category {category}: {len(all_metas)} items, {len(all_reviews)} reviews")
            item_id_to_items: dict[str, dict] = {row["parent_asin"]: row for row in all_metas.iter_rows(named=True)}
            if self.max_items_per_category is not None:
                all_reviews = all_reviews.head(int(self.max_items_per_category))
                print(f"{BYELLOW}Warning: Limiting to {self.max_items_per_category} reviews for debugging.{RST}")
            all_entries = []
            for review in tqdm(
                all_reviews.iter_rows(named=True), desc=f"Creating raw dataset for {category}", total=len(all_reviews)
            ):
                item_id = review["parent_asin"]
                if item_id not in item_id_to_items:
                    print(f"Warning: Item ID {item_id} not found for review from user {review['user_id']}. Skipping.")
                    continue
                item = item_id_to_items[item_id]
                post_prompt = {
                    "role": item["store"],
                    "content": f'{item["title"]}: {item["description"]}',
                    "metadata": json.dumps(item),
                }
                entry = {
                    "prompt": [post_prompt],
                    "completion": f'{review["title"]}: {review["text"]}',  # str
                    "post_id": item_id,  # str: a globally unique id for this post.
                    "user_id": review["user_id"],  # str: a globally unique id for this comment (i.e., the complettion)
                    "timestamp": review["timestamp"],  # int: UTC timestamp
                    "metadata": json.dumps(review),  # metadata for this comment (i.e., the completion).
                }
                all_entries.append(entry)
            print(f"[{datetime.now()}] Memory before converting to Dataset: {memory_usage()}")
            split_size = len(all_entries) // self.category_splits
            for split_idx in tqdm(
                range(self.category_splits), desc=f"Creating splits for {category}", total=self.category_splits
            ):
                now = datetime.now()
                if split_idx < self.category_splits - 1:
                    split_entries = all_entries[split_idx * split_size : (split_idx + 1) * split_size]
                else:
                    split_entries = all_entries[split_idx * split_size :]
                split_dataset = Dataset.from_list(split_entries, features=FEATURES)
                print(f"[{datetime.now()}] Memory after creating split {split_idx}: {memory_usage()}")
                print(
                    f"Category {category} split {split_idx}: converting to Dataset took {datetime.now() - now}, entries: {len(split_dataset)}"
                )
                if self.push_to_hub:
                    split_dataset.push_to_hub(self.push_to_hub, config_name=category, split=f"split{split_idx}", private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Reviews 2023 Dataset Filtering.")
    parser.add_argument("--categories", nargs="+", type=str, required=True, help="List of categories to filter reviews.")
    parser.add_argument(
        "--data_dirname",
        type=str,
        default="data",
        help="Directory to save the preprocessed datasets",
    )
    parser.add_argument(
        "--category_splits",
        type=int,
        default=1,
        help="Number of splits per category (to adapt to low memory). Default is 1 (no splitting).",
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
        help="Path to .env file with necessary credentials (e.g., OpenAI API key)",
    )
    # Debug only
    parser.add_argument(
        "--max_items_per_category",
        type=int,
        default=None,
        help="Specify the maximum number of items to process per category (for debugging only).",
    )
    args = parser.parse_args()

    dataset = AmazonReviewDataset(
        categories=args.categories,
        category_splits=args.category_splits,
        push_to_hub=args.push_to_hub,
        config_path=args.config,
        max_items_per_category=args.max_items_per_category,
    )

    asyncio.run(dataset.create_raw_dataset())
