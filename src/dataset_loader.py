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