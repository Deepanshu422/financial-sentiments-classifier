from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ARTIFACTS_DIR = BASE_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
FINAL_MODEL_DIR = ARTIFACTS_DIR / "final_model"
LOGS_DIR = ARTIFACTS_DIR / "logs"
METRICS_FILE = ARTIFACTS_DIR / "eval_results.json"

for path in [CHECKPOINTS_DIR, FINAL_MODEL_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Model & Dataset specifications

BASE_MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "zeroshot/financial_phrasebank"
DATASET_CONFIG = "sentences_allagree"
DATASET_URL = (
    "https://raw.githubusercontent.com/maxwellsarpong/"
    "NLP-financial-text-processing-dataset/master/Sentences_AllAgree.txt"
)

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
NUM_LABELS = len(ID2LABEL)


FREEZE_BACKBONE = False


# Training Hyperparameters
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
SEED = 42

# Hardware device configuration
DEVICE = "cpu"

