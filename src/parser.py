from tqdm import tqdm
import re
from types import SimpleNamespace
from utils import get_current_dir


args = SimpleNamespace(
    file_path=get_current_dir().parent / "data" / "chat.txt",
)



# def parse_whatsapp_chat(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         pattern = r'\[(.*?)\] (.*?): (.*)'
#         messages = []
#         for line in file:
#             match = re.match(pattern, line)
#             if match:
#                 timestamp, sender, content = match.groups()
#                 messages.append({
#                     'timestamp': timestamp,
#                     'sender': sender,
#                     'content': content
#                 })
#     return messages

def parse_whatsapp_chat(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            pass
    return None

messages = parse_whatsapp_chat(args.file_path)
