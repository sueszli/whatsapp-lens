from pathlib import Path
import json
import glob
from types import SimpleNamespace

import pandas as pd
from langdetect import detect
from tqdm import tqdm

from utils import get_current_dir


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
    
    # replace media messages with placeholder (for freq analysis)
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



def get_partner_name(df: pd.DataFrame, path: Path) -> str:
    df = df.copy()
    authors = df["author"].unique()
    authors = [a for a in authors if a != "server"]
    assert len(authors) == 2

    filename = Path(path).name.split(".")[0]

    fst_sim = sum([1 for c in authors[0].lower() if c in filename.lower()]) / len(authors[0])
    snd_sim = sum([1 for c in authors[1].lower() if c in filename.lower()]) / len(authors[1])

    return authors[0] if fst_sim > snd_sim else authors[1]


def get_frequency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pass

if __name__ == "__main__":
    args = SimpleNamespace(
        inputpath=get_current_dir().parent / "data" / "robustness",
    )

    # for path in glob.glob(str(args.inputpath / "*.csv")):
    path = glob.glob(str(args.inputpath / "*.csv"))[0]

    df = pd.read_csv(path)
    df = preprocess(df)

    language = get_language(df)
    partnername = get_partner_name(df, path)

    # guess gender from author and partner
    # guess age from author and partner

    # drop placeholder messages after frequency analysis


    # get frequency metadata

    # get topics metadata

    # get relationship metadata with latents

    # maybe generate data using llms for training

    # for line in df["message"]:
    #     print(line)
