# use llm to generate data
# ideally each run of this file should generate a new dataset

# 1) generate dialogue between different people using an llm
# 2) generate randomly increasing timestamps using a random walk


from typing import SimpleNamespace
from utils import get_current_dir

args = SimpleNamespace(
    prompt="""
    The following is a conversation between two people. The conversation is about a new partnership between two companies. The first person is named John Doe and the second person is named Jane Doe. They are discussing the terms of the partnership.
    """,
    author_name = "John Doe",
    partner_name = "Jane Doe",
    outputpath = get_current_dir().parent / "data" / "synthetic"
)

# set up two models to talk with eachother


