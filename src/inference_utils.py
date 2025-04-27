import os
import pandas as pd
from evaluate import load
from config import * 


def load_test_cases(full_df_path, test_csv_path):
    full_df = pd.read_csv(full_df_path)
    test_df = pd.read_csv(test_csv_path)
    test_df['patient_id'] = test_df['patient_id'].apply(lambda x: int(str(x).replace('tensor(', '').replace(')', '')))

    test_ids = test_df['patient_id'].tolist()
    test_subset = full_df[full_df['patient_id'].isin(test_ids)].reset_index(drop=True)
    return test_subset


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
