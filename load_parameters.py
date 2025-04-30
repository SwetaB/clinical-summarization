import json

def params(config_path="parameters.json"):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Flatten nested keys for easy access
    flat = {
        "model_name": cfg["model"]["name"],
        "input_file": cfg["data"]["input_file"],
        "dataset_tag": cfg["data"]["dataset_tag"],
        "experiment_name": cfg["data"]["experiment_name"],
        "max_input_length": cfg["training"]["max_input_length"],
        "max_target_length": cfg["training"]["max_target_length"],
        "batch_size": cfg["training"]["batch_size"],
        "epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["learning_rate"],
        "warmup_steps": cfg["training"]["warmup_steps"]
    }

    return flat
