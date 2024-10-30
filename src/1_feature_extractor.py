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

    # preprocess
    df = df[df["author"] != "server"]
    df = df[~df["message"].str.contains("New messages will disappear from this chat 24 hours after they're sent, except when kept. Tap to change.")]
    df = df[~df["message"].str.contains("<Media omitted>")]  # media
    df = df[~df["message"].str.contains("(file attached)")]  # media
    df = df[~df["message"].str.contains("location: https://maps.google.com/?q")]  # media
    poll_mask = df["message"].str.contains("|".join(["POLL:", "OPTION:", "votes"]), case=False)  # polls
    df = df[~poll_mask]

    assert len(df["author"].unique()) == 2
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


if __name__ == "__main__":
    args = SimpleNamespace(
        inputpath=get_current_dir().parent / "data" / "robustness",
    )

    # for path in glob.glob(str(args.inputpath / "*.csv")):
    path = glob.glob(str(args.inputpath / "*.csv"))[0]

    df = pd.read_csv(path)
    df = preprocess(df)

    lang = get_language(df)
    print(lang)

    # get frequency metadata

    # get topics metadata

    # get relationship metadata with latents

    # maybe generate data using llms for training

    # for line in df["message"]:
    #     print(line)
