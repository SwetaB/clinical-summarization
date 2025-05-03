import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch
from datasets import load_dataset
from weighted_trainer import WeightedChunkTrainer, TrainerWithTrainLoss

from src.data.load import DataLoading
from src.data.preprocess import PreprocessData
from src.data.split import DataSplit

from model_manager import ModelManager
from trainer_manager import TrainManager

from utils import *
from config import *
from load_parameters import params



def check_saved_splits(base_path):
    required_dirs = ["train", "val", "test"]
    return all(os.path.isdir(os.path.join(base_path, d)) for d in required_dirs)


# Function to run the entire process
def run_fine_tuning(file_path, **kwargs):
   
    # load data
    data_loader = DataLoading(file_path)
    dataset = data_loader.load_data()

    # Load Model, Tokenizer
    try:
        model_loader = ModelManager(model_path=MODEL_NAME)
        model, tokenizer = model_loader.model, model_loader.tokenizer
    except Exception as e:
        print(f"Error occurred while loading model/tokenizer: {e}")
        return
    
    try:
        if check_saved_splits(TRAIN_TEST_SPLIT_DIR):
            print("Splits found. Loading Data..")
            train_dataset = load_from_disk(os.path.join(TRAIN_TEST_SPLIT_DIR, "train"))
            val_dataset = load_from_disk(os.path.join(TRAIN_TEST_SPLIT_DIR, "val"))
        else:
            # Tokenize the data
            print("Splits not found. Starting Tokenizing and Splitting")
            processor = PreprocessData(
                    tokenizer=tokenizer, id_column="patient_id",
                    input_column="case", output_column="structured_summary", 
                    max_input_length=512, max_target_length=256,
                    chunking=True)

            tokenized_data = processor.map_tokenized_data(dataset=dataset, tokenize_fn=processor.tokenize_fn,
                                                        remove_columns=["case", "structured_summary"])

            data_splitter = DataSplit(tokenized_data)
            train_dataset, val_dataset = data_splitter.split_train_test(return_test=False)

            print(f"Tokenization done. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    except Exception as e:
        print(f"Error occured during tokenization: {e}")
        return

    # Fine-tune the model
    try:
        trainer = TrainManager(model, tokenizer, train_dataset, val_dataset, save_dir=MODEL_OUTPUT_DIR,
                               training_args=kwargs, trainer_class=TrainerWithTrainLoss)
        trainer.train()
    except Exception as e:
        print(f"Error occured during training: {e}")
        return
    
    print(f"Fine-tuning complete. Model saved at: {MODEL_OUTPUT_DIR}/final_model")


# Run the fine-tuning process
if __name__ == "__main__":

    parameter_values = params("parameters.json")

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default=parameter_values['input_file'], help="Path to input training file")
    parser.add_argument('--batch_size', type=int, default=parameter_values['batch_size'], help="Training batch size per device")
    parser.add_argument('--epochs', type=int, default=parameter_values['epochs'], help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=parameter_values['learning_rate'], help="Learning rate")
    parser.add_argument('--warmup_steps', type=int, default=parameter_values['warmup_steps'], help="Warmup steps for scheduler")
    args = parser.parse_args()


    # Training parameters
    training_kwargs = {
        "num_train_epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps
    }

    run_fine_tuning(args.input_file, **training_kwargs)