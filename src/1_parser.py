import csv
import glob
import os
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

from tqdm import tqdm

from utils import get_current_dir, set_seed

set_seed()


def parse_whatsapp_chat(inputpath: Path, outputpath: Path) -> None:
    writer = csv.writer(open(outputpath, "w", newline="", encoding="utf-8"))
    writer.writerow(["timestamp", "author", "message"])

    current_message: Optional[Dict[str, str]] = None

    with open(inputpath, "r", encoding="utf-8") as inputfile:
        for line in tqdm(inputfile):
            line = line.strip()
            if not line:  # skip empty
                continue

            pattern = r"(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s(?:([^:]+):\s)?(.+)"
            match = re.match(pattern, line)
            if match:
                if current_message:
                    writer.writerow([current_message["timestamp"], current_message["author"], current_message["message"]])

                timestamp_str, author, message = match.groups()
                assert timestamp_str and message
                current_message = {
                    "timestamp": datetime.strptime(timestamp_str, "%m/%d/%y, %H:%M").strftime("%Y-%m-%d %H:%M:%S"),
                    "author": author.strip() if author else "server",
                    "message": message.strip(),
                }
            elif current_message:
                current_message["message"] += "\n" + line

    if current_message:  # last message
        writer.writerow([current_message["timestamp"], current_message["author"], current_message["message"]])


def validate_csv(path: Path) -> None:
    content_iter = csv.reader(open(path, "r", encoding="utf-8"))
    header = next(content_iter)
    assert header == ["timestamp", "author", "message"]

    prev_timestamp = None
    for row in content_iter:
        assert len(row) == 3  # three cols
        assert (not prev_timestamp) or (prev_timestamp < row[0])  # increasing timestamps


if __name__ == "__main__":
    args = SimpleNamespace(
        inputpath=get_current_dir().parent / "data" / "personal",
    )
    os.makedirs(args.inputpath, exist_ok=True)

    for path in glob.glob(str(args.inputpath / "*.txt")):
        print(f"processing: {Path(path).name}")
        outputpath = args.inputpath / f"{Path(path).stem}.csv"
        parse_whatsapp_chat(Path(path), outputpath)

        validate_csv(outputpath)
