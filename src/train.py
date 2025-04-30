import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from weighted_trainer import WeightedChunkTrainer, TrainerWithTrainLoss
from utils import *
from config import *
from load_parameters import params

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

    return dataset


def train_model(train_dataset, val_dataset, tokenizer, model, save_dir="./models/clinical_summarization_checkpoints",
                **kwargs):
    '''
    # When using distributed training, ensure only the main process saves the model
    # trainer.is_world_process_zero() can be used if saving within the Trainer logic
    # If saving outside, a simple check might be needed depending on setup
    # For this example, saving after trainer.train() is fine as Trainer handles this
    '''

    num_train_epochs = kwargs.get('num_train_epochs', 10)
    per_device_batch_size = kwargs.get('per_device_batch_size', 4)
    learning_rate = kwargs.get('learning_rate', 5e-5)
    warmup_steps = kwargs.get('warmup_steps', 500)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,    # Batch size per GPU/CPU for training. Total batch size = per_device_train_batch_size * num_gpus
        per_device_eval_batch_size=per_device_batch_size,  
        learning_rate=learning_rate,  
        warmup_steps=warmup_steps,         # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                 # strength of weight decay
        logging_dir='./logs',              # directory for storing logs
        eval_strategy="epoch",             # evaluation strategy
        save_strategy="epoch",             # save checkpoint every epoch
        save_total_limit=3,                # limit the total number of checkpoints
        load_best_model_at_end=True,       # load the best model when finished training
        metric_for_best_model="eval_loss", # Metric to monitor for best model
        greater_is_better=False,           # For loss, lower is better
        logging_strategy="epoch",
        logging_first_step=True,
        report_to="none"              
    )

    # Use the data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = TrainerWithTrainLoss(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
        eval_dataset=val_dataset,            
        tokenizer=tokenizer,                 
        data_collator=data_collator          
    )

    print("Starting model training...")
    trainer.train()

    final_save_dir = os.path.join(save_dir, "final_model")
    os.makedirs(final_save_dir, exist_ok=True)
    trainer.save_model(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"Model saved to {final_save_dir}")

    print("Saving model training metrics")
    training_metrics = trainer.state.log_history
    pd.DataFrame(training_metrics).to_csv(os.path.join(save_dir, "training_metrics.csv"), index=False)

    return trainer


# Function to run the entire process
def run_fine_tuning(file_path, **kwargs):
   
    # load data
    df_processed = pd.read_csv(file_path)
    
    # Check if train, test, data exists
    if os.path.exists(TRAIN_TEST_SPLIT_DIR):
        expected_dirs = [
            os.path.join(TRAIN_TEST_SPLIT_DIR, d)
            for d in os.listdir(TRAIN_TEST_SPLIT_DIR)
            if d in ("train", "val", "test") and os.path.isdir(os.path.join(TRAIN_TEST_SPLIT_DIR, d))
        ]
    else:
        expected_dirs = []

    try:
        model, tokenizer = load_model_and_tokenizer(model_path=MODEL_NAME)
    except Exception as e:
        print(f"Error occurred while loading model/tokenizer: {e}")
        return

    # Tokenize the data
    print("Starting data tokenization...")
    try:
        if len(expected_dirs) == 3:
            train_dataset = load_dataset(os.path.join(TRAIN_TEST_SPLIT_DIR, "train"))
            val_dataset = load_dataset(os.path.join(TRAIN_TEST_SPLIT_DIR, "val"))
            test_dataset = load_dataset(os.path.join(TRAIN_TEST_SPLIT_DIR, "test"))
        else:
            tokenized_data = tokenize_data(df_processed, tokenizer, max_input_length=512, max_target_length=256)
            train_dataset, val_dataset, test_dataset = split_train_test(tokenized_data)
            save_datasets(train_dataset, val_dataset, test_dataset, model_name=MODEL_NAME, dataset_tag=DATASET_TAG)

        print(f"Tokenization done. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    except Exception as e:
        print(f"Error occured during tokenization: {e}")
        return

    # Fine-tune the model
    try:
        trainer = train_model(train_dataset, val_dataset, tokenizer, model, 
                              save_dir=MODEL_OUTPUT_DIR, **kwargs)
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