import os
import argparse
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .utils import load_model_and_tokenizer, get_model_save_dir
from .inference_utils import (load_test_cases, compute_rouge, compute_bertscore, 
                             save_metrics_csv, save_predictions)
from .config import *


def generate_summary(model, tokenizer, text_or_texts, max_input_length=512, base_output_length=128, device="cpu"):
    def estimate_output_length(input_text):
        token_count = len(tokenizer.encode(input_text, truncation=False))
        if token_count > 700:
            return 512
        elif token_count > 400:
            return 256
        else:
            return base_output_length

    if isinstance(text_or_texts, list):
        output_lengths = [estimate_output_length(t) for t in text_or_texts]
        inputs = tokenizer(text_or_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_length).to(device)
        with torch.no_grad():
            outputs = []
            for idx, input_ids in enumerate(inputs["input_ids"]):
                input_batch = {
                    "input_ids": input_ids.unsqueeze(0),
                    "attention_mask": inputs["attention_mask"][idx].unsqueeze(0)
                }
                generated = model.generate(**input_batch, max_length=output_lengths[idx])
                outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
        return outputs
    else:
        output_len = estimate_output_length(text_or_texts)
        inputs = tokenizer(text_or_texts, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=output_len)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run clinical summarization inference")
    parser.add_argument('--input_filename', type=str, required=True, help="Path to the input CSV file for inference")
    parser.add_argument('--split', type=str, choices=["train", "val", "test"], default="val", help="Data split type (default=val)")
    args = parser.parse_args()
    
    inference_file_path = f"{TRAIN_TEST_SPLIT_DIR}/{args.input_filename}"

    model, tokenizer = load_model_and_tokenizer(model_path=FINAL_MODEL_DIR)
    test_dataset = load_test_cases(INPUT_FILE, inference_file_path)

    test_texts = test_dataset['case'].tolist()
    patient_ids=test_dataset['patient_id'].to_list()
    
    # Run inference
    generated_summaries = generate_summary(model, tokenizer, test_texts, base_output_length=256)

    save_predictions(patient_ids=patient_ids, 
                     inputs=test_texts, 
                     predictions=generated_summaries,
                     model_name=MODEL_NAME, 
                     dataset_tag=DATASET_TAG,
                     experiment_name=EXPERIMENT_NAME,
                     split_type=args.split)
    
    rouge_scores = compute_rouge(generated_summaries, test_texts)
    bertscore_scores = compute_bertscore(generated_summaries, test_texts)

    # Merge the metrics you care about
    metrics_to_save = {
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore_f1": bertscore_scores["bertscore_f1"]
    }

    # Save to CSV
    save_metrics_csv(metrics_to_save, 
                    model_name=MODEL_NAME, 
                    dataset_tag=DATASET_TAG, 
                    experiment_name=EXPERIMENT_NAME,
                    split_type=args.split)
