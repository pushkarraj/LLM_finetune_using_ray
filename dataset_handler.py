from transformers import AutoTokenizer

from datasets import load_dataset
import ray.data
import numpy as np
import pandas as pd
import os



print("Loading tiny_shakespeare dataset")
current_dataset = load_dataset("tiny_shakespeare")
print(current_dataset)

ray_datasets = ray.data.from_huggingface(current_dataset)


def split_text(batch: pd.DataFrame) -> pd.DataFrame:
    text = list(batch["text"])
    flat_text = "".join(text)
    split_text = [
        x.strip()
        for x in flat_text.split("\n")
        if x.strip() and not x.strip()[-1] == ":"
    ]
    return pd.DataFrame(split_text, columns=["text"])


def tokenize(batch: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["text"]),
        truncation=True,
        max_length=block_size,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)

# splitter = BatchMapper(split_text, batch_format="pandas")
# tokenizer = BatchMapper(tokenize, batch_format="pandas")