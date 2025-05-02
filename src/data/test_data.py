from datasets import Dataset
from transformers import AutoTokenizer
import pandas as pd
from src.data.preprocess import PreprocessData
from src.data.load import DataLoading

# Load dummy data
data_loader = DataLoading("data/summaries/structured_summaries_20250425_202927_checkpoint_500.csv")
dataset = data_loader.load_data()

# Tokenizer (can be any compatible model)
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# PreprocessData
processor = PreprocessData(
    tokenizer=tokenizer,
    id_column="patient_id",
    input_column="case",
    output_column="structured_summary",
    max_input_length=512,
    max_target_length=256,
    chunking=True
)

# Run tokenization
tokenized_dataset = processor.map_tokenized_data(
    dataset=dataset,
    tokenize_fn=processor.tokenize_fn,
    remove_columns=["case", "structured_summary"]
)

# Check
print(tokenized_dataset)
print(tokenized_dataset[0])
