import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pandas as pd
from langdetect import detect
from tqdm import tqdm

from utils import get_current_dir, get_device

weightspath = get_current_dir().parent / "weights"
os.makedirs(weightspath, exist_ok=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # type conversion
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df["author"] = df["author"].astype("category")
    df["message"] = df["message"].astype(str)

    # drop server messages
    df = df[df["author"] != "server"]
    df = df[~df["message"].str.contains("New messages will disappear from this chat 24 hours after they're sent, except when kept. Tap to change.")]
    assert len(df["author"].unique()) == 2

    # replace media messages with placeholder
    placeholder = "<MEDIA OMITTED>"
    df["message"] = df["message"].str.replace("<Media omitted>", placeholder)
    df["message"] = df["message"].str.replace("(file attached)", placeholder)
    df["message"] = df["message"].str.replace("location: https://maps.google.com/?q", placeholder)
    poll_mask = df["message"].str.contains("|".join(["POLL:", "OPTION:", "votes"]), case=False)  # polls
    df.loc[poll_mask, "message"] = placeholder
    return df


def get_language(df: pd.DataFrame) -> str:
    df = df.copy()

    rnd_indices = df.sample(100 if len(df) > 100 else len(df)).index  # random samples
    langs = []
    for idx in tqdm(rnd_indices):
        try:
            langs.append(detect(df.loc[idx, "message"]))
        except:
            pass
    lang = max(set(langs), key=langs.count)  # majority vote

    assert lang in ["en", "de"]
    return lang


def get_author_name(df: pd.DataFrame, path: Path) -> str:
    df = df.copy()

    authors = df["author"].unique()
    assert len(authors) == 2

    filename = Path(path).name.split(".")[0]
    sim0 = sum([1 for c in authors[0].lower() if c in filename.lower()]) / len(authors[0])
    sim1 = sum([1 for c in authors[1].lower() if c in filename.lower()]) / len(authors[1])
    authorname = authors[1] if sim0 > sim1 else authors[0]
    return authorname


def get_gender(name: str) -> Optional[str]:
    from transformers import pipeline

    device = get_device(disable_mps=False)
    sex_classifier = pipeline("text-classification", model="padmajabfrl/Gender-Classification", device=device, model_kwargs={"cache_dir": weightspath})
    cls = sex_classifier(name)[0]["label"]
    if cls not in ["Male", "Female"]:
        cls = None
    return "m" if cls == "Male" else "f"


def get_freq_stats(df: pd.DataFrame, authorname: str) -> dict:
    df = df.copy()

    total_messages = len(df)
    author_messages = df[df["author"] == authorname]
    partner_messages = df[df["author"] != authorname]
    return {
        "author_message_ratio": float(len(author_messages) / total_messages),
        "partner_message_ratio": float(len(partner_messages) / total_messages),
        "author_avg_word_count": float(author_messages["message"].str.split().apply(len).mean()),
        "partner_avg_word_count": float(partner_messages["message"].str.split().apply(len).mean()),
        "author_media_count": int(author_messages["message"].str.count("<MEDIA OMITTED>").sum()),
        "partner_media_count": int(partner_messages["message"].str.count("<MEDIA OMITTED>").sum()),
        "author_avg_emoji_count": float(author_messages["message"].str.count(r"[\U0001f600-\U0001f650]").mean()),
        "partner_avg_emoji_count": float(partner_messages["message"].str.count(r"[\U0001f600-\U0001f650]").mean()),
    }


if __name__ == "__main__":
    args = SimpleNamespace(
        inputpath=get_current_dir().parent / "data" / "robustness",
    )

    # for path in glob.glob(str(args.inputpath / "*.csv")):
    path = glob.glob(str(args.inputpath / "*.csv"))[0]

    df = pd.read_csv(path)
    df = preprocess(df)

    author_name = get_author_name(df, path)
    partner_name = df["author"].unique()[0] if df["author"].unique()[0] != author_name else df["author"].unique()[1]

    results = {
        "conversation_language": get_language(df),
        "author_name": author_name,
        "partner_name": partner_name,
        "author_gender": get_gender(author_name),
        "partner_gender": get_gender(partner_name),
        **get_freq_stats(df, author_name),
    }
    print(json.dumps(results, indent=4, ensure_ascii=False))

    # get topics metadata

    # get relationship metadata with latents

    # maybe generate data using llms for training

    # for line in df["message"]:
    #     print(line)
