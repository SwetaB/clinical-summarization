import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import os # Import os for checking DeepSpeed config file

# Note: To use DeepSpeed, you need to install it: pip install deepspeed transformers datasets pandas scikit-learn

def save_datasets_as_csv(train_dataset, val_dataset, train_file='data/train_test/train_data-1.csv', val_file='data/train_test/val_data-1.csv'):
    """
    Save the tokenized train and validation datasets to CSV files.
    Note: This saves the token IDs and patient_uid.
    
    Args:
        train_dataset: The tokenized training dataset.
        val_dataset: The tokenized validation dataset.
        train_file (str): The name of the file to save the training data.
        val_file (str): The name of the file to save the validation data.
    """
    # Convert to pandas DataFrame
    train_df = pd.DataFrame({
        'patient_uid': train_dataset['patient_uid'],
        'input_ids': train_dataset['input_ids'],
        'attention_mask': train_dataset['attention_mask'],
        'labels': train_dataset['labels']
    })
    val_df = pd.DataFrame({
        'patient_uid': val_dataset['patient_uid'],
        'input_ids': val_dataset['input_ids'],
        'attention_mask': val_dataset['attention_mask'],
        'labels': val_dataset['labels']
    })

    # Save to CSV
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)

    print(f"Training data saved to {train_file}")
    print(f"Validation data saved to {val_file}")


def load_model_and_tokenizer(model_path: str):
    """
    Load the model and tokenizer for fine-tuning or inference.
    """
    print(f"Attempting to load model and tokenizer from: {model_path}")
    try:
        # Using T5 as Falconsai/medical_summarization is based on T5
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        raise OSError(f"Error loading model or tokenizer from {model_path}. Details: {e}")


def tokenize_data(df, tokenizer):
    """
    Tokenize the data using the provided tokenizer.
    Modified to remove padding='max_length' for dynamic padding and preserve patient_uid.
    
    Args:
        df (pandas.DataFrame): The dataframe with preprocessed text. Must contain 'patient_uid',
                               'Processed_Patient_Text', and 'Processed_Abstract' columns.
        tokenizer: The tokenizer (e.g., from Hugging Face).
    
    Returns:
        Tuple: tokenized train dataset, tokenized validation dataset, list of train uids, list of val uids.
    """
    # Convert to Hugging Face Dataset, including patient_uid
    dataset = Dataset.from_pandas(df[['patient_uid','Processed_Patient_Text', 'Processed_Abstract']])
    print("Original Dataset structure:", dataset)

    max_input_length = 512
    max_target_length = 250

    def tokenize_function(examples):
        return tokenizer(examples['Processed_Patient_Text'], truncation=True, max_length=max_input_length)

    dataset = dataset.map(tokenize_function, batched=True)

    def tokenize_target(examples):
        examples['Processed_Abstract'] = [str(text) if isinstance(text, str) else "" for text in examples['Processed_Abstract']]
        encoding = tokenizer(examples['Processed_Abstract'], truncation=True, max_length=max_target_length)
        examples['labels'] = encoding['input_ids']
        return examples

    dataset = dataset.map(tokenize_target, batched=True)

    dataset = dataset.remove_columns(['Processed_Patient_Text', 'Processed_Abstract'])

    print(f"Columns after tokenization and removing original text: {dataset.column_names}") 

    dataset.set_format(type='torch', columns=['patient_uid', 'input_ids', 'attention_mask', 'labels'])
    
    split_datasets = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_datasets['train']
    val_dataset = split_datasets['test']

    train_uids = train_dataset['patient_uid']
    val_uids = val_dataset['patient_uid']

    # save_datasets_as_csv(train_dataset, val_dataset) # Uncomment if you need to save tokenized data

    return train_dataset, val_dataset, train_uids, val_uids


def train_model(train_dataset, val_dataset, tokenizer, model, deepspeed_config_path):
    """
    Fine-tune the model using the training and validation datasets with DeepSpeed.
    
    Args:
        train_dataset: The tokenized training dataset (must include patient_uid).
        val_dataset: The tokenized validation dataset (must include patient_uid).
        tokenizer: The tokenizer to use for the model.
        model: The pre-trained model.
        deepspeed_config_path (str): Path to the DeepSpeed configuration JSON file.
    
    Returns:
        Trainer: The trained model.
    """

    # Check if the DeepSpeed config file exists
    if not os.path.exists(deepspeed_config_path):
        raise FileNotFoundError(f"DeepSpeed config file not found at: {deepspeed_config_path}")
        
    # Define the training arguments
    # DeepSpeed configuration is added via the 'deepspeed' argument
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",

        # --- DeepSpeed Argument ---
        # Provide the path to your DeepSpeed configuration JSON file
        deepspeed=deepspeed_config_path,

        # --- Mixed Precision (often configured in DeepSpeed config) ---
        # You can enable fp16/bf16 here or within the DeepSpeed config.
        # If using DeepSpeed, it's common to configure it in the JSON.
        # fp16=True, # Example: Enable FP16 if not in DeepSpeed config
        # bf16=False, # Example: Enable BF16 if not in DeepSpeed config

        # --- Multi-GPU Arguments (handled by DeepSpeed launch) ---
        # You typically don't need to set local_rank or data_parallel_backend
        # when using the deepspeed launch command.
        # local_rank=-1, # Set by the launch utility
        # data_parallel_backend="nccl", # Handled by DeepSpeed
    )

    # Initialize the Data Collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Initialize the Trainer
    # The Trainer automatically uses the DeepSpeed configuration provided in training_args
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting model training with DeepSpeed...")
    trainer.train()
    print("Training finished.")

    return trainer


def save_model(model, tokenizer, output_dir):
    """
    Save the fine-tuned model and tokenizer.
    
    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer used for preprocessing.
        output_dir (str): Directory to save the model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # When using DeepSpeed, saving is handled differently.
    # The Trainer's save_model method correctly handles DeepSpeed saving.
    # If you were saving manually outside the Trainer, you'd use:
    # model.save_checkpoint(output_dir)
    # But with Trainer, model.save_pretrained is often sufficient,
    # or the Trainer's own save_model logic is used.
    # Let's use the Trainer's built-in save method via trainer.save_model()
    # or rely on save_strategy="epoch" and load_best_model_at_end=True
    # which save checkpoints using DeepSpeed's methods.

    # To save the final model explicitly after training finishes:
    # Ensure this is only called by the main process if not using Trainer's save_model
    # if trainer.is_world_process_zero(): # Check if running in distributed context
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


def run_fine_tuning(file_path, deepspeed_config_path):
    """
    Load data, preprocess it, tokenize, fine-tune the model with DeepSpeed, and save it.
    
    Args:
        file_path (str): Path to the raw data.
        deepspeed_config_path (str): Path to the DeepSpeed configuration JSON file.
    """
    try:
        df_processed = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
        print(f"Data shape: {df_processed.shape}")
        print("Data columns:", df_processed.columns)
        if 'patient_uid' not in df_processed.columns or 'Processed_Patient_Text' not in df_processed.columns or 'Processed_Abstract' not in df_processed.columns:
             raise ValueError("DataFrame must contain 'patient_uid', 'Processed_Patient_Text', and 'Processed_Abstract' columns.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return
    except ValueError as e:
        print(f"Error loading or validating data: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return

    MODEL_PATH = 'Falconsai/medical_summarization'
    try:
        model, tokenizer = load_model_and_tokenizer(model_path=MODEL_PATH)
    except OSError as e:
        print(f"Could not load model or tokenizer: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading model/tokenizer: {e}")
        return

    print("Starting data tokenization...")
    try:
        train_dataset, val_dataset, train_uids, val_uids = tokenize_data(df_processed, tokenizer)
        print("Data tokenization finished.")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Number of training UIDs: {len(train_uids)}")
        print(f"Number of validation UIDs: {len(val_uids)}")
    except Exception as e:
        print(f"An error occurred during data tokenization: {e}")
        return

    try:
        # Pass the deepspeed config path to the training function
        trainer = train_model(train_dataset, val_dataset, tokenizer, model, deepspeed_config_path)
    except FileNotFoundError as e:
        print(f"DeepSpeed config error: {e}")
        return
    except Exception as e:
        print(f"An error occurred during model training with DeepSpeed: {e}")
        return

    MODEL_OUTPUT_DIR='./models/fine_tuned_Falconsai_MedicalSumm_deepspeed'
    try:
        # The Trainer's save_model method correctly handles DeepSpeed saving
        # It saves the model and optimizer states according to the DeepSpeed config
        # and ensures only the main process saves.
        trainer.save_model(MODEL_OUTPUT_DIR)
        # Save tokenizer separately as trainer.save_model might not save it depending on version/config
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        print(f"Model and tokenizer saved to {MODEL_OUTPUT_DIR}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        return


# Run the fine-tuning process
if __name__ == "__main__":
    file_path = 'data/processed/PMC-Patients-Processed-1.csv'
    # --- DeepSpeed Configuration ---
    # You need to create a JSON file with your DeepSpeed configuration.
    # Example: deepspeed_config.json
    # {
    #     "zero_optimization": {
    #         "stage": 2, # or 3 for more memory savings
    #         "offload_optimizer": {
    #             "device": "cpu",
    #             "pin_memory": true
    #         },
    #         "offload_param": {
    #             "device": "cpu",
    #             "pin_memory": true
    #         }
    #     },
    #     "fp16": {
    #         "enabled": true # Enable mixed precision
    #     },
    #     "train_batch_size": "auto", # Let DeepSpeed calculate based on per_device_train_batch_size and num_gpus
    #     "train_micro_batch_size_per_gpu": 8, # This should match or be derived from per_device_train_batch_size
    #     "gradient_accumulation_steps": "auto", # Let DeepSpeed calculate based on train_batch_size and micro_batch_size
    #     "gradient_clipping": 1.0 # Example: Add gradient clipping
    # }
    #
    # Adjust the 'stage' and 'offload' settings based on your memory constraints and
    # the size of the model/data. Stage 2 is common, Stage 3 provides maximum memory savings.
    #
    # --- Launching with DeepSpeed ---
    # You typically launch this script using the `deepspeed` command:
    # deepspeed --num_gpus your_num_gpus your_script_name.py --deepspeed_config deepspeed_config.json
    #
    # Or using `accelerate launch` with a DeepSpeed config specified in the Accelerate config:
    # accelerate launch your_script_name.py --deepspeed_config deepspeed_config.json
    #
    # The --deepspeed_config argument is how you pass the config path to your script.
    # We will read this argument in the script.

    # --- How to get the deepspeed_config_path from command line arguments ---
    import sys
    deepspeed_config_path = None
    # Parse command line arguments to find --deepspeed_config
    if "--deepspeed_config" in sys.argv:
        deepspeed_config_path = sys.argv[sys.argv.index("--deepspeed_config") + 1]
    else:
        # Provide a default path or raise an error if the argument is required
        # For this example, let's assume the config is in the current directory
        # You should adjust this based on where you save your config file
        print("Warning: --deepspeed_config not found in arguments. Assuming default path './deepspeed_config.json'")
        deepspeed_config_path = './deepspeed_config.json' # Default path

    # Ensure the config path was found/provided
    if deepspeed_config_path is None:
         print("Error: DeepSpeed configuration file path not provided. Use --deepspeed_config <path>")
         sys.exit(1)


    # Run the fine-tuning process with the DeepSpeed config path
    run_fine_tuning(file_path, deepspeed_config_path)

