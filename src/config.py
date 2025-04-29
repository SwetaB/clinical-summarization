import torch
import os
# config.py

# Base paths
BASE_MODEL_DIR = "./models"
BASE_DATA_DIR = "./data/train_test"
BASE_OUTPUT_DIR = "./outputs"

# Model and dataset info
MODEL_NAME = os.getenv("MODEL_NAME", "t5-small")
DATASET_TAG = os.getenv("DATASET_TAG", "clinical_notes_30000")
EXPERIMENT_NAME=os.getenv("EXPERIMENT_NAME", "t5_small_30000_run1")

# INPUT FILE PATH
INPUT_FILE = os.getenv("INPUT_FILE", './data/summaries/structured_summaries_20250425_202927_checkpoint_30000.csv')

# Common settings
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Model save folder (automatically consistent)
MODEL_FOLDER = MODEL_NAME.replace("/", "_")
TRAIN_TEST_SPLIT_DIR = f"{BASE_DATA_DIR}/{MODEL_FOLDER}/{DATASET_TAG}"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/{MODEL_FOLDER}/{DATASET_TAG}"
FINAL_MODEL_DIR = f"{BASE_MODEL_DIR}/{MODEL_FOLDER}/{DATASET_TAG}/final_model"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

