import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import os
import sys # Import sys to use sys.exit() for cleaner exits

# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path: str):
    """
    Load the model and tokenizer for inference.

    Args:
        model_path (str): Path to the fine-tuned model or a Hugging Face model ID.

    Returns:
        model: The loaded model.
        tokenizer: The tokenizer corresponding to the model.
    """
    print(f"Attempting to load model and tokenizer from: {model_path}")

    # Basic check if the model_path looks like a local directory path
    # This is a heuristic; a Hugging Face ID might coincidentally look like a path.
    is_local_path = os.path.exists(model_path)

    if is_local_path:
         if not os.path.isdir(model_path):
             raise FileNotFoundError(f"Model path exists but is not a directory: {model_path}")
         # You might want to check for specific files like config.json, pytorch_model.bin, etc.
         # For simplicity, we'll rely on from_pretrained raising an error if files are missing.

    try:
        # from_pretrained can handle both local paths and Hugging Face IDs
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        # Provide a more informative error message if loading fails
        raise OSError(f"Error loading model or tokenizer from {model_path}. "
                      f"Ensure the path is correct and contains valid model files, "
                      f"or that the Hugging Face ID is correct. Details: {e}")

# Function to perform inference (summarization) on a given text
def generate_summary(text: str, model, tokenizer, max_length: int = 150, num_beams: int = 4) -> str:
    """
    Generate a summary for the input text using the fine-tuned model.

    Args:
        text (str): The input text to summarize.
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.
        max_length (int): Maximum length of the summary.
        num_beams (int): Number of beams for beam search during generation.

    Returns:
        str: The generated summary.
    """
    # Add the "summarize: " prefix for BART model if needed.
    # This depends on how the model was fine-tuned. If your fine-tuning
    # data did NOT use this prefix, remove the following line.
    input_text = "summarize: " + text

    # Tokenize the input text
    # return_tensors="pt" for PyTorch tensors
    # truncation=True to handle texts longer than the model's max input length (typically 1024 for BART)
    # padding=True to pad shorter texts to the longest sequence in the batch or model max length if batch size is 1
    # max_length is set to 1024, which is common for BART input.
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding=True)

    # Determine the device (GPU or CPU) and move the model and inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Using device: {device}") # Indicate which device is being used

    # Generate the summary IDs
    # max_length here is for the generated summary output length.
    # num_beams controls the beam search width.
    # early_stopping=True means generation stops once all beam candidates have reached the end token or max_length.
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=num_beams, early_stopping=True)

    # Decode the summary back to text
    # skip_special_tokens=True removes special tokens added during tokenization (like <s>, </s>, <pad>).
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Remove the "summarize: " prefix from the generated summary if it appears.
    if summary.startswith("summarize: "):
         summary = summary[len("summarize: "):].strip()
    else:
         summary = summary.strip() # Just strip whitespace if no prefix generated

    return summary


# Function to run the inference for multiple texts
def run_inference(input_texts: list[str], model, tokenizer) -> list[str]:
    """
    Perform inference (summarization) for multiple input texts.

    Args:
        input_texts (list): A list of input texts to summarize.
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.

    Returns:
        list: A list of generated summaries. Each element is either a summary string or an error message.
    """
    summaries = []
    if not input_texts:
        print("No input texts provided for summarization.")
        return []

    print(f"\nStarting summarization for {len(input_texts)} text(s)...")
    for i, text in enumerate(input_texts):
        print(f"\nProcessing text {i+1}/{len(input_texts)}...")
        # Adding basic error handling per text item
        if not isinstance(text, str) or not text.strip():
            print(f"Skipping text {i+1}: Input is not a valid string or is empty.")
            summaries.append(f"Error: Invalid or empty input text for item {i+1}")
            continue # Skip to the next text

        try:
            # Add a simple check or truncation for very long input texts if needed
            # This helps prevent potential issues with extremely large inputs
            if len(text) > 5000: # Example limit
                 print(f"Warning: Text {i+1} is very long ({len(text)} characters). It will be truncated by the tokenizer.")

            summary = generate_summary(text, model, tokenizer)
            summaries.append(summary)
            print(f"Finished processing text {i+1}.")
        except Exception as e:
            print(f"Error processing text {i+1}: {e}")
            # Append an error message to the summaries list instead of crashing
            summaries.append(f"Error summarizing text {i+1}: {e}")
    print("\nSummarization complete.")
    return summaries

# Example usage
if __name__ == "__main__":
    # Define the path to your fine-tuned model or a Hugging Face model ID.
    # Replace this with the actual path where your model is saved or the ID.
    # Example local path: 'models/fine_tuned_bart'
    # Example Hugging Face ID: 'bart-large-cnn'
    MODEL_PATH = 'models/fine_tuned_bart'

    # Load the model and tokenizer
    # The load_model_and_tokenizer function now includes error handling.
    try:
        model, tokenizer = load_model_and_tokenizer(model_path=MODEL_PATH)
    except (FileNotFoundError, OSError) as e:
        print(f"Failed to load model: {e}")
        print("Please ensure the MODEL_PATH is correct.")
        sys.exit(1) # Exit the script with a non-zero status code indicating an error

    # Sample patient texts (or input data). Replace with your actual data.
    patient_texts = [
        "This patient is a 55-year-old male who was admitted to the hospital for chest pain, shortness of breath, and elevated heart rate. Medical history includes hypertension and type 2 diabetes. Medications include Lisinopril and Metformin. Physical examination showed mild edema in the lower extremities. Labs revealed elevated troponin levels.",
        "Hey", # A very short text to see how the model handles it
        "The patient presented with symptoms including severe headache, nausea, and sensitivity to light. No previous history of migraines was reported. Diagnostic tests are underway to rule out other conditions. Medication has been administered to alleviate the immediate symptoms. Follow-up appointment scheduled for next week.",
        "", # An empty string
        123 # An invalid input type
        # Add more texts here if needed
    ]

    # Run inference
    summaries = run_inference(patient_texts, model, tokenizer)

    # Print out the summaries
    print("\n" + "="*30)
    print("--- GENERATED SUMMARIES ---")
    print("="*30 + "\n")

    for i, summary in enumerate(summaries):
        print(f"--- Summary {i+1} ---")
        print(summary)
        print("-" * 20) # Separator for clarity

    print("\n" + "="*30)
    print("--- END OF SUMMARIES ---")
    print("="*30 + "\n")