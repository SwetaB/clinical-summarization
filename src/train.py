import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from datasets import Dataset
from weighted_trainer import WeightedChunkTrainer
from utils import load_model_and_tokenizer, get_model_save_dir, save_datasets, chunk_text
from config import *

def tokenize_data(df, tokenizer, max_input_length=1024, max_target_length=256):
    dataset = Dataset.from_pandas(df[['patient_id', 'case', 'structured_summary']])

    def tokenize_examples(examples):
        all_inputs, all_chunk_ids, all_targets, all_patient_ids= [], [], [], []

        # using all chunks and extending targets to all chunks
        for patient_id, text, summary in zip(examples['patient_id'], examples['case'], examples['structured_summary']):
            chunks = chunk_text(text, tokenizer, max_tokens=max_input_length)
            # all_inputs.append(chunks[0])  # Use only first chunk for now
            all_inputs.extend(chunks)
            all_chunk_ids.extend(list(range(len(chunks))))
            all_targets.extend([summary] * len(chunks))
            all_patient_ids.extend([patient_id] * len(chunks))

        # Why is padding max lenght
        inputs = tokenizer(all_inputs, truncation=True, padding='max_length', max_length=max_input_length)
        labels = tokenizer(all_targets, truncation=True, padding='max_length', max_length=max_target_length)

        inputs['labels'] = labels['input_ids']
        inputs['chunk_id'] = all_chunk_ids
        inputs['patient_id'] = all_patient_ids
        return inputs

    dataset = dataset.map(tokenize_examples, batched=True, remove_columns=['case', 'structured_summary'])

    assert 'patient_id' in dataset.features, "Error: patient_id was dropped during tokenization!"
    print("Patient IDs correctly preserved after tokenization.")
    
    dataset.set_format(type='torch', columns=['patient_id','input_ids', 'attention_mask', 'labels', 'chunk_id'])

    # Split into train, val, test
    train_val_split = dataset.train_test_split(test_size=0.3, seed=42)
    train_dataset = train_val_split['train']
    val_test_split = train_val_split['test'].train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    return train_dataset, val_dataset, test_dataset


def train_model(train_dataset, val_dataset, tokenizer, model, save_dir="./models/clinical_summarization_checkpoints"):
    '''
    # When using distributed training, ensure only the main process saves the model
    # trainer.is_world_process_zero() can be used if saving within the Trainer logic
    # If saving outside, a simple check might be needed depending on setup
    # For this example, saving after trainer.train() is fine as Trainer handles this
    '''

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,    # Batch size per GPU/CPU for training. Total batch size = per_device_train_batch_size * num_gpus
        per_device_eval_batch_size=4,    
        warmup_steps=500,                 # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                # strength of weight decay
        logging_dir='./logs',             # directory for storing logs
        eval_strategy="epoch",            # evaluation strategy
        save_strategy="epoch",            # save checkpoint every epoch
        save_total_limit=3,               # limit the total number of checkpoints
        load_best_model_at_end=True,      # load the best model when finished training
        metric_for_best_model="eval_loss", # Metric to monitor for best model
        greater_is_better=False,         # For loss, lower is better
        logging_steps=100, 
        report_to="none"                 # Disable reporting to external services like W&B
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Trainer(
        model=model,                         # the pre-trained model
        args=training_args,                  # training arguments
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # validation dataset
        tokenizer=tokenizer,                 # tokenizer for preprocessing
        data_collator=data_collator          # Use the data collator for dynamic padding
    )

    print("Starting model training...")
    trainer.train()
    
    final_save_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_save_dir, exist_ok=True)
    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Model saved to {final_save_dir}")

    return trainer


# Function to run the entire process
def run_fine_tuning(file_path):
   
    # load data
    df_processed = pd.read_csv(file_path)
    
    # Load the tokenizer, model
    # MODEL_PATH = 'Falconsai/medical_summarization'
    # MODEL_PATH = 'facebook/bart-large-cnn'
    # MODEL_PATH = "sshleifer/distilbart-cnn-12-6"

    # MODEL_PATH="t5-small"
    SAVE_DIR = get_model_save_dir(model_name=MODEL_NAME, dataset_tag=DATASET_TAG)

    try:
        model, tokenizer = load_model_and_tokenizer(model_path=MODEL_NAME)
    except Exception as e:
        print(f"Error occurred while loading model/tokenizer: {e}")
        return

    # Tokenize the data
    print("Starting data tokenization...")
    try:
        train_dataset, val_dataset, test_dataset = tokenize_data(df_processed, tokenizer, max_input_length=512, max_target_length=256)
        print(f"Tokenization done. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        save_datasets(train_dataset, val_dataset, test_dataset, model_name=MODEL_NAME, dataset_tag=DATASET_TAG)
    except Exception as e:
        print(f"Error occured during tokenization: {e}")
        return

    # Fine-tune the model
    try:
        trainer = train_model(train_dataset, val_dataset, tokenizer, model, save_dir=SAVE_DIR)
    except Exception as e:
        print(f"Error occured during training: {e}")
        return
    
    print(f"Fine-tuning complete. Model saved at: {SAVE_DIR}/final_model")


# Run the fine-tuning process
if __name__ == "__main__":
    run_fine_tuning(INPUT_FILE)