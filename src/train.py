import json
import logging
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from src.config import (
    BASE_MODEL_NAME,
    NUM_LABELS,
    ID2LABEL,
    LABEL2ID,
    CHECKPOINTS_DIR,
    FINAL_MODEL_DIR,
    LOGS_DIR,
    METRICS_FILE,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    FREEZE_BACKBONE,
)
from src.dataset_loader import prepare_data
from src.metrics import compute_metrics

# configure logging to see clean status messages in console
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_model():
    """
    Loads pre-trained DistilBERT and attaches a fresh 3-class classification head.
    Optionally freezes base transformer layers depending on config.
    """

    logger.info(f"Loading base model: {BASE_MODEL_NAME} with {NUM_LABELS} target classes")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if FREEZE_BACKBONE:
        logger.info("Freezing DistilBERT backbone parameters (feature extraction mode)...")

        # setting param.requires_grad = False, To turnoff gradient calculations for model's base layers
        for param in model.distilbert.parameter():
            param.requires_grad = False

        # verify tainable parameters 
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    else:
        logger.info("All layers unfrozen (full fine-tuning mode).")

    return model


def run_training():
    # fetch tokenized dataset splits and tokenizer
    tokenized_dataset, tokenizer = prepare_data()

    # building model architecture
    model = build_model()

    # dynamic batch collator (pads sequences dynamically per batch, not statically)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # setting up training arguments
    # modern versions of transformers support 'eval_strategy'
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINTS_DIR),
        eval_strategy="epoch",             # run evaluation at the end of each epoch
        save_strategy="epoch",             # checkpoint saved to match eval_strategy
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,       # load the highest performing checkpoint when done
        metric_for_best_model="f1",        # Best model is judged by weighted F1, not loss
        greater_is_better=True,
        logging_steps=25,
        save_total_limit=2,                # keep only top 2 checkpoints on disk to save space
        report_to="none",                  # can change to 'wandb' or 'tensorboard' if needed
    )

    # initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # run training loop
    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    # evaluate on held-out test split
    logger.info("Running final evaluation on test set...")
    eval_metrics = trainer.evaluate()
    logger.info(f"Evaluation Results: {json.dumps(eval_metrics, indent=2)}")

    # save production deployment artifacts
    logger.info(f"Saving final model and tokenizer to: {FINAL_MODEL_DIR}")
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(FINAL_MODEL_DIR))
    tokenizer.save_pretrained(str(FINAL_MODEL_DIR))

    # export audit metrics to JSON
    with open(METRICS_FILE, "w") as f:
        json.dump(eval_metrics, f, indent=4)
    logger.info(f"Saved evaluation metrics to {METRICS_FILE}")


if __name__ == "__main__":
    run_training()
