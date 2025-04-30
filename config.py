import torch
import os
from load_parameters import params

parameter_values = params("parameters.json")

# Base paths
BASE_MODEL_DIR = "./models"
BASE_DATA_DIR = "./data/train_test"
BASE_OUTPUT_DIR = "./outputs"

# Model and dataset info
MODEL_NAME = parameter_values["model_name"]
DATASET_TAG = parameter_values['dataset_tag']
EXPERIMENT_NAME = parameter_values['experiment_name']

# Model save folder (automatically consistent)
MODEL_NAME_FOLDER = MODEL_NAME.replace("/", "_")
TRAIN_TEST_SPLIT_DIR = f"{BASE_DATA_DIR}/{MODEL_NAME_FOLDER}/{DATASET_TAG}"
MODEL_OUTPUT_DIR = f"{BASE_MODEL_DIR}/{MODEL_NAME_FOLDER}/{DATASET_TAG}"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/{MODEL_NAME_FOLDER}/{DATASET_TAG}"
FINAL_MODEL_DIR = f"{BASE_MODEL_DIR}/{MODEL_NAME_FOLDER}/{DATASET_TAG}/final_model"

# Train settings
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
