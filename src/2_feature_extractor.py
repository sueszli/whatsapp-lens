import csv
import glob
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import emojis
import numpy as np
import pandas as pd
from langdetect import detect
from tqdm import tqdm

from utils import get_current_dir, get_device, set_seed

set_seed()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

weightspath = get_current_dir().parent / "weights"
os.makedirs(weightspath, exist_ok=True)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("preprocessing")

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


def drop_media(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    placeholder = "<MEDIA OMITTED>"
    df = df[~df["message"].str.contains(placeholder)]
    return df


def get_author_name(df: pd.DataFrame, path: Path) -> str:
    df = df.copy()

    authors = df["author"].unique()
    assert len(authors) == 2

    filename = Path(path).name.split(".")[0]
    sim0 = sum([1 for c in authors[0].lower() if c in filename.lower()]) / len(authors[0])
    sim1 = sum([1 for c in authors[1].lower() if c in filename.lower()]) / len(authors[1])
    authorname = authors[1] if sim0 > sim1 else authors[0]
    return authorname


def get_language(df: pd.DataFrame) -> str:
    print("detecting language")
    df = df.copy()

    rnd_indices = df.sample(100 if len(df) > 100 else len(df)).index  # random samples
    langs = []
    for idx in rnd_indices:
        try:
            langs.append(detect(df.loc[idx, "message"]))
        except:
            pass
    lang = max(set(langs), key=langs.count)  # majority vote

    assert lang in ["en", "de"]
    return lang


def get_gender_stats(df: pd.DataFrame, authorname: str) -> dict:
    print("classifying gender")

    def get_gender(name: str) -> Optional[str]:
        from transformers import pipeline

        sex_classifier = pipeline("text-classification", model="padmajabfrl/Gender-Classification", device=get_device(disable_mps=False), model_kwargs={"cache_dir": weightspath})
        cls = sex_classifier(name)[0]["label"]
        if cls not in ["Male", "Female"]:
            cls = None
        return "m" if cls == "Male" else "f"

    partnername = df["author"].unique()[0] if df["author"].unique()[0] != authorname else df["author"].unique()[1]
    return {
        "author_gender": get_gender(authorname),
        "partner_gender": get_gender(partnername),
    }


def get_monthly_sentiments(df: pd.DataFrame, authorname: str, sample_size: int) -> dict:
    print("sentiment analysis")
    df = df.copy()
    df = drop_media(df)

    def get_monthly_user_sentiments(df: pd.DataFrame, authorname: str, sample_size: int) -> list[float]:
        from transformers import pipeline

        sentiment_classifier = pipeline("sentiment-analysis", device=get_device(disable_mps=False), model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", model_kwargs={"cache_dir": weightspath})

        monthly_sentiments = []

        # cluster by month
        dfm = df.copy()
        dfm = dfm[dfm["author"] == authorname]
        dfm = dfm.set_index("datetime").resample("ME")

        for month, group in dfm:
            # get sampled average sentiment
            sample_size_adjusted = min(sample_size, len(group))
            sampled_messages = group["message"].sample(n=sample_size_adjusted).tolist()

            sentiments = [sentiment_classifier(message)[0]["score"] for message in sampled_messages]
            avg = sum(sentiments) / len(sentiments)
            monthly_sentiments.append(avg)
        return monthly_sentiments

    partnername = df["author"].unique()[0] if df["author"].unique()[0] != authorname else df["author"].unique()[1]
    return {
        "author_monthly_sentiments": get_monthly_user_sentiments(df, authorname, sample_size),
        "partner_monthly_sentiments": get_monthly_user_sentiments(df, partnername, sample_size),
    }


def get_monthly_toxicity(df: pd.DataFrame, authorname: str, sample_size: int) -> dict:
    print("toxicity analysis")
    df = df.copy()
    df = drop_media(df)

    def get_monthly_user_toxicity(df: pd.DataFrame, authorname: str, sample_size: int) -> list[float]:
        from transformers import pipeline

        toxicity_classifier = pipeline("text-classification", device=get_device(disable_mps=False), model="citizenlab/distilbert-base-multilingual-cased-toxicity", model_kwargs={"cache_dir": weightspath})

        monthly_toxicity = []

        # cluster by month
        dfm = df.copy()
        dfm = dfm[dfm["author"] == authorname]
        dfm = dfm.set_index("datetime").resample("ME")

        for month, group in dfm:
            # get sampled average toxicity
            sample_size_adjusted = min(sample_size, len(group))
            sampled_messages = group["message"].sample(n=sample_size_adjusted).tolist()

            toxicities = [toxicity_classifier(message)[0]["label"] == "TOXIC" for message in sampled_messages]
            avg = sum(toxicities) / len(toxicities)
            monthly_toxicity.append(avg)
        return monthly_toxicity

    partnername = df["author"].unique()[0] if df["author"].unique()[0] != authorname else df["author"].unique()[1]
    return {
        "author_monthly_toxicity": get_monthly_user_toxicity(df, authorname, sample_size),
        "partner_monthly_toxicity": get_monthly_user_toxicity(df, partnername, sample_size),
    }


def get_topic_diversity_score(df: pd.DataFrame) -> float:
    print("topic modeling")
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    df = df.copy()
    df = drop_media(df)
    messages = df["message"].tolist()

    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2", device=get_device(disable_mps=False), cache_folder=weightspath)
    vectorizer = CountVectorizer(
        min_df=1,  # minimum document frequency
        max_df=1.0,  # maximum document frequency
        ngram_range=(1, 2),  # allow single words and bigrams
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=3,  # minimum number of messages in a topic
        nr_topics="auto",
        language="multilingual",
    )
    topics, probs = topic_model.fit_transform(messages)  # fit
    topic_info = topic_model.get_topic_info()
    topic_diversity = len(topic_info[topic_info["Topic"] != -1]) / len(messages)
    return topic_diversity


def get_embedding(df: pd.DataFrame) -> list[list[float]]:
    print("embedding generation")
    from sentence_transformers import SentenceTransformer

    df = df.copy()
    df = drop_media(df)
    messages = df["message"].tolist()
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=get_device(disable_mps=False), cache_folder=weightspath)
    embeddings = model.encode(messages)
    return embeddings.tolist()


def get_freq_stats(df: pd.DataFrame, authorname: str) -> dict:
    print("frequency analysis")
    df = df.copy()

    author_messages = df[df["author"] == authorname]
    partner_messages = df[df["author"] != authorname]
    assert len(df["author"].unique()) == 2

    conversation_threshold = 60 * 60 * 2  # after 2 hours, the conversation has ended
    df["time_diff"] = df["datetime"].diff().dt.total_seconds()  # time difference in seconds
    df["new_conversation"] = df["time_diff"] > conversation_threshold  # assign new conversation
    df["conversation_id"] = df["new_conversation"].cumsum()  # assign conversation id

    def get_response_times(messages1, messages2):
        responses = []
        # group by conversation
        for conv_id in df["conversation_id"].unique():
            conv_msgs = df[df["conversation_id"] == conv_id].copy()
            for i in range(len(conv_msgs) - 1):
                # change in author
                if conv_msgs.iloc[i]["author"] in messages1["author"].values and conv_msgs.iloc[i + 1]["author"] in messages2["author"].values:
                    time_diff = (conv_msgs.iloc[i + 1]["datetime"] - conv_msgs.iloc[i]["datetime"]).total_seconds()
                    if time_diff < conversation_threshold:
                        responses.append(time_diff)
        return np.mean(responses) if responses else 0

    return {
        "total_messages": len(df),
        # message ratio
        "author_message_ratio": len(author_messages) / len(df),
        "partner_message_ratio": len(partner_messages) / len(df),
        # message length
        "author_avg_word_count": author_messages["message"].str.split().str.len().mean(),
        "partner_avg_word_count": partner_messages["message"].str.split().str.len().mean(),
        # media type count
        "author_media_count": int(author_messages["message"].str.count("<MEDIA OMITTED>").sum()),
        "partner_media_count": int(partner_messages["message"].str.count("<MEDIA OMITTED>").sum()),
        "author_emoji_count": int(author_messages["message"].apply(emojis.get).str.len().sum()),
        "partner_emoji_count": int(partner_messages["message"].apply(emojis.get).str.len().sum()),
        "author_url_count": int(author_messages["message"].str.count(r"https?://").sum()),
        "partner_url_count": int(partner_messages["message"].str.count(r"https?://").sum()),
        # vocabulary richness
        "author_vocabulary_size": len(set(author_messages["message"].str.split().explode())),
        "partner_vocabulary_size": len(set(partner_messages["message"].str.split().explode())),
        # time between each message
        "total_conversations": df["conversation_id"].nunique(),
        "total_duration_days": (df["datetime"].max() - df["datetime"].min()).days + 1,
        "author_message_freq_s": author_messages["datetime"].diff().mean().total_seconds(),
        "partner_message_freq_s": partner_messages["datetime"].diff().mean().total_seconds(),
        # time to response
        "author_response_time_s": get_response_times(partner_messages, author_messages),
        "partner_response_time_s": get_response_times(author_messages, partner_messages),
        # active time of day
        "author_avg_active_time": author_messages["datetime"].dt.hour.mean(),
        "partner_avg_active": partner_messages["datetime"].dt.hour.mean(),
        # conversation inits
        "author_conversation_initiations": int(sum(df[df["new_conversation"] == True]["author"].value_counts().to_dict().values()) - int(df[df["new_conversation"] == True]["author"].value_counts().to_dict().get(authorname, 0))),
        "partner_conversation_initiations": int(sum(df[df["new_conversation"] == True]["author"].value_counts().to_dict().values()) - int(df[df["new_conversation"] == True]["author"].value_counts().to_dict().get(authorname, 0))),
    }


if __name__ == "__main__":
    args = SimpleNamespace(
        inputpath=get_current_dir().parent / "data" / "robustness",
        outputpath=get_current_dir().parent / "data" / "results",
    )
    os.makedirs(args.outputpath, exist_ok=True)
    os.makedirs(weightspath, exist_ok=True)

    for path in tqdm(glob.glob(str(args.inputpath / "*.csv"))):
        path = glob.glob(str(args.inputpath / "*.csv"))[0]

        df = pd.read_csv(path)
        df = preprocess(df)
        author_name = get_author_name(df, path)

        results = {
            "conversation_language": get_language(df),
            "author_name": author_name,
            "partner_name": df["author"].unique()[0] if df["author"].unique()[0] != author_name else df["author"].unique()[1],
            **get_monthly_sentiments(df, author_name, sample_size=1_000),
            **get_monthly_toxicity(df, author_name, sample_size=1_000),
            **get_gender_stats(df, author_name),
            "topic_diversity": get_topic_diversity_score(df),
            **get_freq_stats(df, author_name),
            "embeddings": get_embedding(df),
        }

        with open(args.outputpath / f"results.csv", "w") as f:
            if os.stat(args.outputpath / f"results.csv").st_size == 0:
                writer = csv.DictWriter(f, fieldnames=results.keys())
                writer.writeheader()
            writer = csv.DictWriter(f, fieldnames=results.keys())
            writer.writerow(results)
        print("done")
