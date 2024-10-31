from types import SimpleNamespace

from utils import get_current_dir

if __name__ == "__main__":
    args = SimpleNamespace(
        prompt="""
        The following is a conversation between two people. The conversation is about a new partnership between two companies. The first person is named John Doe and the second person is named Jane Doe. They are discussing the terms of the partnership.
        """,
        author_name="John Doe",
        partner_name="Jane Doe",
        outputpath=get_current_dir().parent / "data" / "synthetic",
    )

