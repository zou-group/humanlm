# =============================================================================
# MTurk HIT Publisher
# =============================================================================
# Required environment variables (set in your shell or in a .env file):
#   MTURK_KEY          - AWS access key for MTurk sandbox
#   MTURK_SECRET       - AWS secret key for MTurk sandbox
#   MTURK_KEY_LIVE     - AWS access key for MTurk production (live mode only)
#   MTURK_SECRET_LIVE  - AWS secret key for MTurk production (live mode only)
#
# You can create a .env file in this directory (or the project root) with
# these variables. By default, python-dotenv loads .env from the current
# working directory.
# =============================================================================

import argparse
import os

import boto3
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--url_link", required=True, help="The Gradio share URL for the HIT")
parser.add_argument("--reward", default="9", help="The reward amount in dollars (default: 9)")
parser.add_argument("--mturk_region", default="us-east-1", help="The region for mturk (default: us-east-1)")
parser.add_argument("--num_hits", type=int, default=1, help="The number of HITs.")
parser.add_argument("--num_assignments", type=int, default=5,
                    help="The number of times that the HIT can be accepted and completed.")
parser.add_argument("--live_mode", action="store_true",
                    help="Whether to run in live mode with real turkers. This will charge your account money. "
                         "Without this flag, HITs are deployed on the sandbox.")
parser.add_argument("--no_qualification", action="store_true",
                    help="Publish HITs without any qualification requirements (sandbox only).")
parser.add_argument("--cheat", action="store_true",
                    help="Publish HITs with cheat mode (sandbox only).")
args = parser.parse_args()

url_link = args.url_link
reward = args.reward

MTURK_URL = f"https://mturk-requester{'' if args.live_mode else '-sandbox'}.{args.mturk_region}.amazonaws.com"

if args.live_mode:
    MTURK_KEY = os.getenv("MTURK_KEY_LIVE")
    MTURK_SECRET = os.getenv("MTURK_SECRET_LIVE")
else:
    MTURK_KEY = os.getenv("MTURK_KEY")
    MTURK_SECRET = os.getenv("MTURK_SECRET")

mturk = boto3.client(
    "mturk",
    aws_access_key_id=MTURK_KEY,
    aws_secret_access_key=MTURK_SECRET,
    region_name=args.mturk_region,
    endpoint_url=MTURK_URL,
)

print(mturk.get_account_balance())
username = "mturk" if args.live_mode else "mturk-sandbox"

if args.cheat and args.live_mode:
    raise ValueError("You can't use --cheat in live mode.")

if args.no_qualification and args.live_mode:
    raise ValueError("You can't use --no_qualification in live mode.")

print("I have $" + mturk.get_account_balance()['AvailableBalance'] + " in my account")

if args.live_mode:
    estimated_cost = 1.25 * float(reward) * args.num_hits * args.num_assignments
    print(f"You are about to publish {args.num_hits} HITs in live mode. "
          f"This will charge your account ${estimated_cost:.2f}.")
    response = input("Are you sure you want to continue? (yes/no): ")
    if response != "yes":
        print("Exiting without publishing HITs.")
        exit()

live_qualifications = [
    # Master worker
    {
        "QualificationTypeId": "2F1QJWKUDD8XADTFD2Q0G6UTO95ALH",
        "Comparator": "Exists",
        "RequiredToPreview": True,
        "ActionsGuarded": "DiscoverPreviewAndAccept",
    },
    # Location
    {
        "QualificationTypeId": "00000000000000000071",
        "Comparator": "In",
        "LocaleValues": [
            {"Country": "US"},
            {"Country": "CA"},
            {"Country": "GB"},
            {"Country": "AU"},
        ],
        "RequiredToPreview": True,
        "ActionsGuarded": "DiscoverPreviewAndAccept",
    },
    # Number of HITs approved
    {
        "QualificationTypeId": "00000000000000000040",
        "Comparator": "GreaterThanOrEqualTo",
        "IntegerValues": [800],
        "RequiredToPreview": True,
        "ActionsGuarded": "DiscoverPreviewAndAccept",
    },
    # Approval rate
    {
        "QualificationTypeId": "000000000000000000L0",
        "Comparator": "GreaterThanOrEqualTo",
        "IntegerValues": [95],
        "RequiredToPreview": True,
        "ActionsGuarded": "DiscoverPreviewAndAccept",
    },
    # Bad worker exclusion
    {
        "QualificationTypeId": "3O42DSCJPXG0RKZ2IIR1UEBTRWCX6H",
        "Comparator": "DoesNotExist",
        "ActionsGuarded": "DiscoverPreviewAndAccept",
    },
]

sandbox_qualification = [] if args.no_qualification else []

for i in range(args.num_hits):
    cheat_suffix = "&amp;cheat=yes" if args.cheat else ""
    question = f"""
        <ExternalQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2006-07-14/ExternalQuestion.xsd">
            <ExternalURL>{url_link}{cheat_suffix}</ExternalURL>
            <FrameHeight>1000</FrameHeight>
        </ExternalQuestion>
        """

    title = "Evaluate AIs that Simulate Users [Short | Reddit]"
    desc = "Respond to a Reddit post and compare your response with three AI-generated responses."

    new_hit = mturk.create_hit(
        Title=title,
        Description=desc,
        Keywords="AI, simulating user, survey",
        Reward=reward,
        MaxAssignments=args.num_assignments,
        LifetimeInSeconds=60 * 60 * 24 * 1 if args.live_mode else 60 * 60 * 24 * 4,
        AssignmentDurationInSeconds=60 * 60 * 2 if args.live_mode else 60 * 120,
        AutoApprovalDelayInSeconds=60 * 10,
        Question=question,
        QualificationRequirements=live_qualifications if args.live_mode else sandbox_qualification,
    )

print(
    f"HIT Group Link: https://worker{'' if args.live_mode else 'sandbox'}.mturk.com/mturk/preview?groupId="
    + new_hit["HIT"]["HITGroupId"] + f" in {'cheat' if args.cheat else 'normal'} mode"
)
