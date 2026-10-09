"""
tests/test_dataset.py
Unit tests verifying dataset loading, split integrity, and tokenization bounds.
"""

from src.config import MAX_LENGTH
from src.dataset_loader import get_tokenizer, prepare_data


def test_tokenizer_initialization():
    tokenizer = get_tokenizer()
    assert tokenizer is not None
    encoded = tokenizer("Operating revenue increased by 5%", truncation=True)
    assert "input_ids" in encoded
    assert "attention_mask" in encoded


def test_prepare_data_splits_and_columns():
    dataset, tokenizer = prepare_data()

    # Verify both train and test splits exist
    assert "train" in dataset
    assert "test" in dataset

    # Verify standard Hugging Face expected columns
    train_sample = dataset["train"][0]
    assert "input_ids" in train_sample
    assert "attention_mask" in train_sample
    assert "labels" in train_sample

    # Verify truncation limits
    assert len(train_sample["input_ids"]) <= MAX_LENGTH