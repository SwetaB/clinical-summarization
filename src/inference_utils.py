import os
import pandas as pd
from evaluate import load
from datasets import load_from_disk
from config import * 


def load_test_cases(full_df_path, test_csv_path):
    full_df = pd.read_csv(full_df_path)
    test_df = pd.read_csv(test_csv_path)
    test_df['patient_id'] = test_df['patient_id'].apply(lambda x: int(str(x).replace('tensor(', '').replace(')', '')))

    test_ids = test_df['patient_id'].tolist()
    test_subset = full_df[full_df['patient_id'].isin(test_ids)].reset_index(drop=True)
    return test_subset

def load_test_dataset(full_df_path, split_dir, split_name):
    full_df = pd.read_csv(full_df_path)
    
    path = os.path.join(split_dir, split_name)
    dataset = load_from_disk(path)
    dataset.reset_format()
    test_ids = dataset['patient_id']

    test_subset = full_df[full_df['patient_id'].isin(test_ids)].reset_index(drop=True)
    return test_subset

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
    

def save_predictions(patient_ids, inputs, predictions, model_name, dataset_tag, experiment_name="experiment", split_type='val'):

    # Mirror structure: model/dataset
    model_folder = model_name.replace("/", "_")
    save_dir = os.path.join(BASE_OUTPUT_DIR, model_folder, dataset_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Prediction file path
    output_csv_path = os.path.join(save_dir, f"{split_type}_predictions_{experiment_name}.csv")

    # Create dataframe
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "input_text": inputs,
        "predicted_summary": predictions
    })

    # Save to CSV
    df.to_csv(output_csv_path, index=False)

    print(f"Predictions saved to {output_csv_path}")


def compute_rouge(predictions, references):
    rouge = load("rouge")
    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    # Return only F1 scores for simplicity
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"]
    }


def compute_bertscore(predictions, references, lang="en"):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang=lang)

    return {
        "bertscore_precision": sum(results["precision"]) / len(results["precision"]),
        "bertscore_recall": sum(results["recall"]) / len(results["recall"]),
        "bertscore_f1": sum(results["f1"]) / len(results["f1"])
    }


def save_metrics_csv(metrics_dict, model_name, dataset_tag, experiment_name="experiment", split_type="val"):

    # Mirror structure
    model_folder = model_name.replace("/", "_")
    save_dir = os.path.join(BASE_OUTPUT_DIR, model_folder, dataset_tag)
    os.makedirs(save_dir, exist_ok=True)

    # Metrics CSV path
    output_csv_path = os.path.join(save_dir, "metrics_summary.csv")

    # Add experiment name to metrics
    metrics_dict["experiment"] = experiment_name
    metrics_dict["split"] = split_type

    # Save or append
    if os.path.exists(output_csv_path):
        existing = pd.read_csv(output_csv_path)
        new_row = pd.DataFrame([metrics_dict])
        final = pd.concat([existing, new_row], ignore_index=True)
    else:
        final = pd.DataFrame([metrics_dict])

    final.to_csv(output_csv_path, index=False)

    print(f"Metrics saved to {output_csv_path}")
