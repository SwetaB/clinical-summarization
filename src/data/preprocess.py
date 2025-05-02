import os
import pandas as pd
from datasets import Dataset


class PreprocessData():
    def __init__(self, tokenizer, id_column, input_column, output_column, max_input_length, max_target_length, chunking=True):
        self.tokenizer = tokenizer
        self.id_column = id_column
        self.input_column = input_column
        self.target_column = output_column
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.chunking = chunking
    
    def chunk_text(self, text):
        token_ids = self.tokenizer.encode(text, truncation=False)
        chunks = []
        for i in range(0, len(token_ids), self.max_input_length):
            chunk_ids = token_ids[i:i+self.max_input_length]
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks

    def tokenize_fn(self, examples):
        all_inputs, all_ids, chunk_ids = [], [], []

        for i in range(len(examples[self.id_column])):
            text = str(examples[self.input_column][i])

            # Chunk or not
            chunks = self.chunk_text(text) if self.chunking else [text]
            all_inputs.extend(chunks)
            all_ids.extend([examples[self.id_column][i]] * len(chunks))
            chunk_ids.extend(list(range(len(chunks))))

        # Tokenize inputs
        tokenized = self.tokenizer(
            all_inputs,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_attention_mask=True
        )

        # Tokenize targets
        if self.target_column:
            target_col = self.target_column
            targets = [
                str(examples[target_col][i])
                for i in range(len(examples[target_col]))
                for _ in (range(len(self.chunk_text(str(examples[self.input_column][i])))) if self.chunking else [1])
            ]
            label_tokens = self.tokenizer(
                targets,
                truncation=True,
                padding="max_length",
                max_length=self.max_target_length
            )
            tokenized["labels"] = label_tokens["input_ids"]

        tokenized[self.id_column] = all_ids
        if self.chunking:
            tokenized["chunk_id"] = chunk_ids
        return tokenized
    
    def map_tokenized_data(self, dataset: Dataset, tokenize_fn, remove_columns: list, extra_columns: list = []) -> Dataset:
        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=remove_columns)

        assert self.id_column in dataset.features, f"Error: {self.id_column} was dropped during tokenization!"
        print(f"{self.id_column} correctly preserved after tokenization.")
        
        if self.chunking:
            dataset.set_format(type='torch', columns=[self.id_column, 'input_ids', 'attention_mask', 'labels', 'chunk_id'] + extra_columns)
        else:
            dataset.set_format(type='torch', columns=[self.id_column, 'input_ids', 'attention_mask', 'labels'] + extra_columns)

        return dataset

def save_tokenized_data(self, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    dataset.save_to_disk(save_dir)
    print(f"Tokenized dataset saved to: {save_dir}")

   