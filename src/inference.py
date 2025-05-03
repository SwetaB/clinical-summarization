import os
import argparse
import pandas as pd
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from inference_utils import *
from config import *
from load_parameters import params
from model_manager import ModelManager


def run_inference(model, tokenizer, evaluation_set):

    test_texts = evaluation_set['case'].tolist()
    patient_ids = evaluation_set['patient_id'].to_list()

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


if __name__ == "__main__":

    parameter_values = params("parameters.json")

    parser = argparse.ArgumentParser(description="Run clinical summarization inference")
    parser.add_argument('--input_filename', type=str,  help="Optional: custom Hugging Face split dir (e.g., 'val')")
    parser.add_argument('--split', type=str, choices=["train", "val", "test"], default="val", help="Data split type (default=val)")
    args = parser.parse_args()
    
    split_name = args.input_filename if args.input_filename else args.split
    inference_dataset = load_test_dataset(parameter_values['input_file'], TRAIN_TEST_SPLIT_DIR, split_name)
    model_loader = ModelManager(model_path=FINAL_MODEL_DIR)
    model, tokenizer = model_loader.model, model_loader.tokenizer

    run_inference(model, tokenizer, inference_dataset)
    
    