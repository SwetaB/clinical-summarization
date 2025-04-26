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


def save_datasets_as_csv(train_dataset, val_dataset,  model_name, dataset_tag):
    base_dir = "./data/train_test"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_folder = model_name.replace("/", "_")
    save_dir = os.path.join(base_dir, model_folder)
    os.makedirs(save_dir, exist_ok=True)

    train_file = os.path.join(save_dir, f"train_{dataset_tag}_{timestamp}.csv")
    val_file = os.path.join(save_dir, f"val_{dataset_tag}_{timestamp}.csv")

    pd.DataFrame(train_dataset).to_csv(train_file, index=False)
    pd.DataFrame(val_dataset).to_csv(val_file, index=False)

    print(f"Training data saved to {train_file}")
    print(f"Validation data saved to {val_file}")

def load_model_and_tokenizer(model_path: str, model_type: str = "auto"):
    print(f"Loading {model_type.upper()} model and tokenizer from: {model_path}")
    try:
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
        print("Successfully loaded.")
        return model, tokenizer
    except Exception as e:
        raise OSError(f"Error loading model/tokenizer from {model_path}. Details: {e}")

def chunk_text(text, tokenizer, max_tokens=1024):
    token_ids = tokenizer.encode(text, truncation=False)
    chunks = []
    for i in range(0, len(token_ids), max_tokens):
        chunk_ids = token_ids[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks


def tokenize_data(df, tokenizer, max_input_length=1024, max_target_length=250):
    dataset = Dataset.from_pandas(df[['patient_id', 'case', 'structured_summary']])

    def tokenize_inputs(examples):
        all_inputs = []
        for text in examples['case']:
            chunks = chunk_text(text, tokenizer, max_tokens=max_input_length)
            all_inputs.append(chunks[0])  # Use only first chunk for now
        return tokenizer(all_inputs, truncation=True, padding='max_length', max_length=max_input_length)

    def tokenize_targets(examples):
        examples['structured_summary'] = [str(t) if isinstance(t, str) else "" for t in examples['structured_summary']]
        targets = tokenizer(examples['structured_summary'], truncation=True, padding='max_length', max_length=max_target_length)
        examples['labels'] = targets['input_ids']
        return examples

    dataset = dataset.map(tokenize_inputs, batched=True)
    dataset = dataset.map(tokenize_targets, batched=True)
    dataset = dataset.remove_columns(['case', 'structured_summary'])
    dataset.set_format(type='torch', columns=['patient_id', 'input_ids', 'attention_mask', 'labels'])

    split = dataset.train_test_split(test_size=0.2, seed=42)
    return split['train'], split['test']


def train_model(train_dataset, val_dataset, tokenizer, model, save_dir="./models/clinical_summarization_checkpoints"):
    '''
    # When using distributed training, ensure only the main process saves the model
    # trainer.is_world_process_zero() can be used if saving within the Trainer logic
    # If saving outside, a simple check might be needed depending on setup
    # For this example, saving after trainer.train() is fine as Trainer handles this
    '''

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=5,             # number of training epochs
        per_device_train_batch_size=8,   # Batch size per GPU/CPU for training. Total batch size = per_device_train_batch_size * num_gpus
        per_device_eval_batch_size=8,    
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        evaluation_strategy="epoch",     # evaluation strategy
        save_strategy="epoch",           # save checkpoint every epoch
        save_total_limit=3,              # limit the total number of checkpoints
        load_best_model_at_end=True,     # load the best model when finished training
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

    MODEL_PATH = "sshleifer/distilbart-cnn-12-6"
    SAVE_DIR = "./models/fine_tuned_model"

    try:
        model, tokenizer = load_model_and_tokenizer(model_path=MODEL_PATH)
    except Exception as e:
        print(f"Error occurred while loading model/tokenizer: {e}")
        return

    # Tokenize the data
    print("Starting data tokenization...")
    try:
        train_dataset, val_dataset = tokenize_data(df_processed, tokenizer)
        print(f"Tokenization done. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        save_datasets_as_csv(train_dataset, val_dataset, model_name=MODEL_PATH, dataset_tag="clinical_notes_16500")
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
    file_path = 'data/summaries/structured_summaries_20250425_202927_checkpoint_16500.csv'  # Adjust path if necessary
    run_fine_tuning(file_path)