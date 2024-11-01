"""
synthetic data generation works great, if you have sufficient compute.
these models are small enough to run on a laptop - but they still generate reasonable responses.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from random import randint, random
from types import SimpleNamespace

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_current_dir, get_device, set_seed

device = get_device(disable_mps=False)
set_seed()
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


weightspath = get_current_dir().parent / "weights"
os.makedirs(weightspath, exist_ok=True)

device = get_device(disable_mps=False)


model_name_1 = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name_2 = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer_1 = AutoTokenizer.from_pretrained(model_name_1, cache_dir=weightspath)
tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2, cache_dir=weightspath)

model_1 = AutoModelForCausalLM.from_pretrained(model_name_1, cache_dir=weightspath).to(device)
model_2 = AutoModelForCausalLM.from_pretrained(model_name_2, cache_dir=weightspath).to(device)


def generate_response(model, tokenizer, prompt, persona):
    context = f"You are {persona}. Respond to: {prompt}"

    inputs = tokenizer(context, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.9,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(context, "").strip()
    response = re.sub(r"[^.!?]*$", "", response)
    return response


def get_next_timestamp(prev_timestamp: datetime):
    assert isinstance(prev_timestamp, datetime)
    delta = timedelta(seconds=randint(1, 600))  # between 1s and 10m
    if random() < 0.01:  # 1% chance of increasing between 1h and 1d
        delta += timedelta(seconds=randint(3600, 86400))
    return prev_timestamp + delta


def store_msg(filepath: Path, name: str, message: str, timestamp: datetime):
    wa_msg = f"{timestamp.strftime('%m/%d/%y, %H:%M')} - {name}: {message}"  # whatsapp format
    with open(filepath, "a") as f:
        f.write(wa_msg + "\n")


if __name__ == "__main__":
    args = SimpleNamespace(
        outputpath=get_current_dir().parent / "data" / "synthetic" / f"WhatsApp Chat with Jane Doe.txt",
    )

    name_1 = "Jane Doe"
    persona_1 = "a book-loving introvert who enjoys reading, coffee shops, and discussing literature. Use literary references, book-related emoji, and mention reading or favorite books occasionally."

    name_2 = "John Smith"
    persona_2 = "a fitness enthusiast who loves working out, healthy eating, and outdoor activities. Use casual language, fitness-related emoji, and occasionally mention workouts or healthy foods."

    init_msg = f"Hey Jane! How's your day going? 🌟"

    iterations = 25

    currtime = datetime.now()
    conversation_history = init_msg
    store_msg(args.outputpath, name_1, init_msg, currtime)
    for i in tqdm(range(iterations)):
        response = generate_response(model_1, tokenizer_1, conversation_history, persona_2)
        store_msg(args.outputpath, name_1, response, get_next_timestamp(currtime))
        conversation_history = response

        response = generate_response(model_2, tokenizer_2, conversation_history, persona_1)
        store_msg(args.outputpath, name_2, response, get_next_timestamp(currtime))
        conversation_history = response
