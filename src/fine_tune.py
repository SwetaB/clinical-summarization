import torch
from transformers import Trainer, TrainingArguments
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import DataCollatorForSeq2Seq 
from datasets import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def save_datasets_as_csv(train_dataset, val_dataset, train_file='data/train_test/train_data-Falconsai_MedicalSumm_1.csv', val_file='data/train_test/val_data-Falconsai_MedicalSumm_1.csv'):
    """
    Save the tokenized train and validation datasets to CSV files.
    
    Args:
        train_dataset: The tokenized training dataset.
        val_dataset: The tokenized validation dataset.
        train_file (str): The name of the file to save the training data.
        val_file (str): The name of the file to save the validation data.
    """
    # Convert to pandas DataFrame
    train_df = pd.DataFrame(train_dataset)
    val_df = pd.DataFrame(val_dataset)

    # Save to CSV
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print(f"Training data saved to {train_file}")
    print(f"Validation data saved to {val_file}")


def load_model_and_tokenizer(model_path: str):
    """
    Load the model and tokenizer for inference.
    """
    print(f"Attempting to load model and tokenizer from: {model_path}")
    try:
        # *** Changed from Bart to T5 ***
        # model = T5ForConditionalGeneration.from_pretrained(model_path)
        # tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        # The error message from the library is quite informative here, keep it
        raise OSError(f"Error loading model or tokenizer from {model_path}. Details: {e}")


def tokenize_data(df, tokenizer):
    """
    Tokenize the data using the provided tokenizer.
    
    Args:
        df (pandas.DataFrame): The dataframe with preprocessed text.
        tokenizer: The tokenizer (e.g., from Hugging Face).
    
    Returns:
        Tuple: tokenized train and validation datasets.
    """

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[['patient_uid', 'Processed_Patient_Text', 'Processed_Abstract']])

    # Define max lengths based on typical model limits and potential data analysis
    # You should analyze your data to confirm these are appropriate
    max_input_length = 512
    max_target_length = 250

    # Tokenize the input (patient text)
    def tokenize_function(examples):
        return tokenizer(examples['Processed_Patient_Text'], truncation=True, max_length=max_input_length)

    dataset = dataset.map(tokenize_function, batched=True)

    # Tokenize the target (abstract)
    def tokenize_target(examples):
        # Ensure that the target (abstract) is a valid string
        # Replace NaN or None with an empty string or placeholder
        examples['Processed_Abstract'] = [str(text) if isinstance(text, str) else "" for text in examples['Processed_Abstract']]
        encoding = tokenizer(examples['Processed_Abstract'], truncation=True, padding='max_length', max_length=max_target_length)
        examples['labels'] = encoding['input_ids']
        return examples


    dataset = dataset.map(tokenize_target, batched=True)

    print(f"Columns after tokenization: {dataset.column_names}") 

    # Remove the original text columns as they are no longer needed for training input
    dataset = dataset.remove_columns(['Processed_Patient_Text', 'Processed_Abstract'])

    # Set format for model input
    dataset.set_format(type='torch', columns=['patient_uid', 'input_ids', 'attention_mask', 'labels'])

    # Split into training and validation sets
    split_datasets = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']

    # Extract patient_uids for train and validation sets
    train_uids = train_dataset['patient_uid']
    val_uids = val_dataset['patient_uid']
    save_datasets_as_csv(train_uids, val_uids)

    # Split into training and validation sets
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.2, seed=42).values()
    
    return train_dataset, val_dataset


# Function to fine-tune the BART model
def train_model(train_dataset, val_dataset, tokenizer, model):
    """
    Fine-tune the BART model using the training and validation datasets.
    
    Args:
        train_dataset: The tokenized training dataset.
        val_dataset: The tokenized validation dataset.
        tokenizer: The tokenizer to use for the model.
        model: The pre-trained model.
    
    Returns:
        Trainer: The trained model.
    """

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./models/fine_tune_facebook/clinical_summarization_checkpoints",',          # output directory
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

    # Initialize the Data Collator for dynamic padding
    # This collator will pad sequences within each batch
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Initialize the Trainer
    trainer = Trainer(
        model=model,                         # the pre-trained model
        args=training_args,                  # training arguments
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # validation dataset
        tokenizer=tokenizer,                 # tokenizer for preprocessing
        data_collator=data_collator          # Use the data collator for dynamic padding
    )

    # Fine-tune the model
    # The trainer.train() call will automatically distribute training across available GPUs
    # when the script is launched correctly.
    print("Starting model training...")
    trainer.train()

    return trainer


# Function to save the fine-tuned model
def save_model(model, tokenizer, output_dir):
    """
    Save the fine-tuned model and tokenizer.
    
    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer used for preprocessing.
        output_dir (str): Directory to save the model.
    """

    os.makedirs(output_dir, exist_ok=True)


    # When using distributed training, ensure only the main process saves the model
    # trainer.is_world_process_zero() can be used if saving within the Trainer logic
    # If saving outside, a simple check might be needed depending on setup
    # For this example, saving after trainer.train() is fine as Trainer handles this
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


# Function to run the entire process
def run_fine_tuning(file_path):
    """
    Load data, preprocess it, tokenize, fine-tune the model, and save it.
    
    Args:
        file_path (str): Path to the raw data.
    """
    # Preprocess the data
    df_processed = pd.read_csv(file_path)
    
    # Load the tokenizer, model
    # example : BartTokenizer.from_pretrained('facebook/bart-base')
    # MODEL_PATH = 'Falconsai/medical_summarization'
    MODEL_PATH = 'facebook/bart-large-cnn'
    try:
        model, tokenizer = load_model_and_tokenizer(model_path=MODEL_PATH)
    except OSError as e:
        print(f"Could not load model or tokenizer: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading model/tokenizer: {e}")
        return

     # Tokenize the data
    print("Starting data tokenization...")
    try:
        # Tokenization happens on the main process, then the dataset is shared# Tokenization happens on the main process, then the dataset is shared
        train_dataset, val_dataset = tokenize_data(df_processed, tokenizer)
        print("Data tokenization finished.")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    except Exception as e:
        print(f"An error occurred during data tokenization: {e}")
        return

    # Fine-tune the model
    try:
        # The train_model function will handle the multi-GPU training via the Trainer
        trainer = train_model(train_dataset, val_dataset, tokenizer, model)
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return

    # Save the model
    MODEL_OUTPUT_DIR='./models/fine_tune_facebook'
    try:
        # The Trainer ensures that save_model is only called by the main process
        save_model(trainer.model, tokenizer, MODEL_OUTPUT_DIR)
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        return


# Run the fine-tuning process
if __name__ == "__main__":
    file_path = 'data/processed/PMC-Patients-Processed.csv'  # Adjust path if necessary
    run_fine_tuning(file_path)