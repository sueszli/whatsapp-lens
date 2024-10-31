# use personal data for final stage
# then make interactive visualizations for github pages

from types import SimpleNamespace
from pathlib import Path
import os
from utils import get_current_dir

if __name__ == "__main__":
    args = SimpleNamespace(
        inputpath=get_current_dir().parent / "data" / "personal" / "results.csv",
    )

    # for path in tqdm(glob.glob(str(args.inputpath / "*.csv"))):
    #     path = glob.glob(str(args.inputpath / "*.csv"))[0]

    #     df = pd.read_csv(path)
    #     df = preprocess(df)
    #     author_name = get_author_name(df, path)

    #     results = {
    #         "conversation_language": get_language(df),
    #         "author_name": author_name,
    #         "partner_name": df["author"].unique()[0] if df["author"].unique()[0] != author_name else df["author"].unique()[1],
    #         **get_monthly_sentiments(df, author_name, sample_size=1_000),
    #         **get_monthly_toxicity(df, author_name, sample_size=1_000),
    #         **get_gender_stats(df, author_name),
    #         "topic_diversity": get_topic_diversity_score(df),
    #         **get_freq_stats(df, author_name),
    #         "embeddings": get_embedding(df),
    #     }

    #     with open(args.outputpath / f"results.csv", "a") as f:
    #         if os.stat(args.outputpath / f"results.csv").st_size == 0:
    #             writer = csv.DictWriter(f, fieldnames=results.keys())
    #             writer.writeheader()
    #         else:
    #             writer = csv.DictWriter(f, fieldnames=results.keys())
    #         writer.writerow(results)
