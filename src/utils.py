import os
import pandas as pd
import torch
import ast
import re
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, 
                           T5Tokenizer, T5ForConditionalGeneration, 
                           BartTokenizer, BartForConditionalGeneration)
from datasets import load_from_disk
from config import *


def load_model_and_tokenizer(model_path: str, model_type: str = "auto"):
    try:
        print(f"Loading {model_type.upper()} model and tokenizer from: {model_path}")
        if model_type.lower() == "bart":
            model = BartForConditionalGeneration.from_pretrained(model_path)
            tokenizer = BartTokenizer.from_pretrained(model_path)
        elif model_type.lower() == "t5":
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            tokenizer = T5Tokenizer.from_pretrained(model_path)
        elif model_type.lower() == "auto":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        model.eval()
        print("Successfully loaded model and tokenizer.")
        return model, tokenizer
    except Exception as e:
        raise OSError(f"Error loading model/tokenizer from {model_path}. Details: {e}")


def get_model_save_dir(model_name, dataset_tag):
    model_folder = model_name.replace("/", "_")
    save_dir = os.path.join(BASE_MODEL_DIR, model_folder, dataset_tag)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def load_dataset(file_path):
    dataset = load_from_disk(file_path)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset


def split_train_test(tokenized_dataset):
    train_val_split = tokenized_dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = train_val_split['train']
    val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    return train_dataset, val_dataset, test_dataset


def save_datasets(train_dataset, val_dataset, test_dataset, model_name, dataset_tag):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_folder = model_name.replace("/", "_")
    save_dir = os.path.join(BASE_DATA_DIR, model_folder, dataset_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Save full datasets to disk (Hugging Face format)
    train_dataset.save_to_disk(os.path.join(save_dir, "train"))
    val_dataset.save_to_disk(os.path.join(save_dir, "val"))
    test_dataset.save_to_disk(os.path.join(save_dir, "test"))

    # train_file = os.path.join(save_dir, f"train_{timestamp}.csv")
    # val_file = os.path.join(save_dir, f"val_{timestamp}.csv")
    # test_file = os.path.join(save_dir, f"test_{timestamp}.csv")

    # pd.DataFrame(train_dataset).to_csv(train_file, index=False)
    # pd.DataFrame(val_dataset).to_csv(val_file, index=False)
    # pd.DataFrame(test_dataset).to_csv(test_file, index=False)

    print(f"Training data saved to")
    print(f"Validation data saved")
    print(f"Test data saved to")


def chunk_text(text, tokenizer, max_tokens=1024):
    token_ids = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

