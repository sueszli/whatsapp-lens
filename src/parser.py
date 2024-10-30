import csv
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

from tqdm import tqdm

from utils import get_current_dir

args = SimpleNamespace(
    inputpath=get_current_dir().parent / "data" / "chat.txt",
    outputpath=get_current_dir().parent / "data" / "chat.csv",
)


def parse_whatsapp_chat(inputpath: Path, outputpath: Path) -> None:
    print(f"processing {inputpath}")
    writer = csv.writer(open(outputpath, "w", newline="", encoding="utf-8"))
    writer.writerow(["timestamp", "author", "message"])

    current_message: Optional[Dict[str, str]] = None

    with open(inputpath, "r", encoding="utf-8") as inputfile:
        for line in tqdm(inputfile):
            line = line.strip()
            if not line:  # skip empty
                continue

            match = re.match(r"(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2})\s-\s(?:([^:]+):\s)?(.+)", line)
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


messages = parse_whatsapp_chat(args.inputpath, args.outputpath)
