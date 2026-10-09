# from datasets import load_dataset
# from transformers import AutoTokenizer
# from src.config import BASE_MODEL_NAME, DATASET_NAME, DATASET_CONFIG, MAX_LENGTH, SEED, DATASET_URL

# def get_tokenizer():
#     # load matching tokenizer for base DistilBERT model
#     return AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# def prepare_data():

#     print(f"[INFO] Fetching dataset: {DATASET_URL})...")
#     raw_data = load_dataset("parquet", data_files={"train": DATASET_URL}, split="train")

#     # Detect the text column name ('sentence' or 'text')
#     text_col = "sentence" if "sentence" in raw_data.column_names else "text"

#     # 80% for training model & 20% for testing
#     dataset_splits = raw_data.train_test_split(test_size=0.2, seed=SEED)

#     tokenizer = get_tokenizer()

#     def tokenize_batch(batch):

#         return tokenizer(
#             batch[text_col],
#             truncation=True,
#             max_length=MAX_LENGTH,
#             padding=False
#         )

#     print("[INFO] Converting text into tokens...")

#     # map tokenization over all sentences in batches
#     tokenized_data = dataset_splits.map(
#         tokenize_batch,
#         batched=True,
#         desc="Tokenizing",
#         # trust_remote_code=True,
#     )

#     # Hugging Face models specifically expect the target column to be named 'labels'
#     if "label" in tokenized_data["train"].column_names:
#         tokenized_data = tokenized_data.rename_column("label", "labels")
    
#     return tokenized_data, tokenizer


# if __name__ == "__main__":
#     # check if we run this script directly
#     data, tok = prepare_data()
#     print("[SUCCESS] Data split ready:")
#     print(f" - Train samples: {len(data['train'])}")
#     print(f" - Test samples:  {len(data['test'])}")
#     print(f" - Example output: {data['train'][0]}")

import io
import urllib.request
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from src.config import (
    BASE_MODEL_NAME,
    DATASET_URL,
    LABEL2ID,
    MAX_LENGTH,
    SEED,
)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL_NAME)


def load_raw_dataframe() -> pd.DataFrame:
    """
    Downloads the Sentences_AllAgree text file and parses
    the 'sentence@sentiment' records into a clean DataFrame.
    """
    print(f"[INFO] Fetching raw dataset from: {DATASET_URL}")
    req = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req) as response:
        content = response.read().decode("latin-1")

    sentences = []
    labels = []

    for line in content.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        # Split from right side once: text@label
        sentence, label_str = line.rsplit("@", 1)
        sentence = sentence.strip()
        label_str = label_str.strip().lower()

        if label_str in LABEL2ID:
            sentences.append(sentence)
            labels.append(LABEL2ID[label_str])

    df = pd.DataFrame({"sentence": sentences, "labels": labels})
    print(f"[INFO] Successfully loaded {len(df)} financial records.")
    return df


def prepare_data():
    # load clean DataFrame
    df = load_raw_dataframe()

    # convert to Hugging Face Dataset and perform 80/20 train/test split
    hf_dataset = Dataset.from_pandas(df)
    dataset_splits = hf_dataset.train_test_split(test_size=0.2, seed=SEED)

    # Tokenize sequences
    tokenizer = get_tokenizer()

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    print("[INFO] Tokenizing dataset...")
    tokenized_data = dataset_splits.map(
        tokenize_batch,
        batched=True,
        desc="Tokenizing",
    )

    return tokenized_data, tokenizer


if __name__ == "__main__":
    data, tok = prepare_data()
    print("[SUCCESS] Data split ready:")
    print(f" - Train samples: {len(data['train'])}")
    print(f" - Test samples:  {len(data['test'])}")
    print(f" - Sample output: {data['train'][0]}")