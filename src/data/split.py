from datasets import Dataset
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config import *


class DataSplit:
    def __init__(self, tokenized_data: Dataset):
        self.tokenized_data = tokenized_data
    
    def split_train_test(self, test_size: float = 0.3, val_size: float = 0.5, save_data: bool = True, return_test: bool = True):
        train_val_split = self.tokenized_data.train_test_split(test_size=test_size, seed=42)
        train_dataset = train_val_split['train']
        val_test_split = train_val_split['test'].train_test_split(test_size=val_size, seed=42)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']

        if save_data:
            self.save_datasets(train_dataset, val_dataset, test_dataset)

        if return_test:
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, val_dataset
    
    def save_datasets(self, train_dataset, val_dataset, test_dataset):
        save_dir = os.path.join(BASE_DATA_DIR, MODEL_NAME_FOLDER, DATASET_TAG)
        os.makedirs(save_dir, exist_ok=True)

        # Save full datasets to disk (Hugging Face format)
        train_dataset.save_to_disk(os.path.join(save_dir, "train"))
        val_dataset.save_to_disk(os.path.join(save_dir, "val"))
        test_dataset.save_to_disk(os.path.join(save_dir, "test"))

        print(f"Training/Testing/Validation data saved to {save_dir}")
