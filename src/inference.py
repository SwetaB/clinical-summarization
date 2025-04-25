import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer


# Function to load the model and tokenizer
def load_model_and_tokenizer(model_path: str):
    """
    Load the model and tokenizer for inference.
    """
    print(f"Attempting to load model and tokenizer from: {model_path}")
    try:
        # *** Changed from Bart to T5 ***
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        # The error message from the library is quite informative here, keep it
        raise OSError(f"Error loading model or tokenizer from {model_path}. Details: {e}")

# Function to perform inference (summarization) on a given text
def generate_summary(text, model, tokenizer, max_length=150, num_beams=4):
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

    # Add the "summarize: " prefix for BART model (if needed)
    input_text = "summarize: " + text
    print(f"Input Text for Summarization: {input_text}")  # Debugging: Check input text

    # Tokenize the input text using the tokenizer
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

    # Generate the summary using the model
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=num_beams, early_stopping=True)

    # Decode the summary back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Remove the "summarize: " prefix from the generated summary if it appears
    summary = summary.replace("summarize: ", "").strip()
    
    return summary


# Function to run the inference for multiple texts
def run_inference(input_texts, model, tokenizer):
    """
    Perform inference (summarization) for multiple input texts.
    
    Args:
        input_texts (list): A list of input texts to summarize.
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.
        
    Returns:
        list: A list of generated summaries.
    """
    summaries = []
    for text in input_texts:
        summary = generate_summary(text, model, tokenizer)
        summaries.append(summary)
    return summaries

# Example usage
if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path='Falconsai/medical_summarization')  # Use your saved model path here
    
    # Sample patient texts (or input data)
    patient_texts = [
        # "This patient is a 55-year-old male who was admitted to the hospital for chest pain, shortness of breath, and elevated heart rate...",
        # "Patient is a 68-year-old male admitted with acute exacerbation of chronic obstructive pulmonary disease (COPD). Presented to the emergency department with increased shortness of breath, cough with yellow sputum, and wheezing for the past three days. History includes smoking 1 pack/day for 40 years, hypertension, and coronary artery disease. Medications: Albuterol inhaler PRN, Symbicort BID, Lisinopril 10mg daily. Physical exam: tachycardic (HR 110), tachypneic (RR 24), decreased breath sounds bilaterally with expiratory wheezing. SaO2 88% on room air. Chest X-ray shows hyperinflation but no consolidation. ABGs: pH 7.32, pCO2 65, pO2 55. Started on IV steroids, antibiotics, and supplemental oxygen via nasal cannula to maintain SaO2 > 90%.",
        "A 52-year-old man underwent acupuncture and cupping treatment at an illegal Chinese medicine clinic for neck and back discomfort. Multiple 0.25 mm × 75 mm needles were utilized and the acupuncture points were located in the middle and on both sides of the upper back and the middle of the lower back. The acupuncture and subsequent cupping treatment lasted 30 minutes, respectively. The patient presented to the hospital with severe gasp and dyspnea about 30 hours later. Physical examinations were as follows: blood pressure (BP) was 149/94 mm Hg, heart rate (HR) was 86 beats/min, and blood oxygen saturation level was 54%. The patient was lucid, was gasping, and had apnea and low respiratory murmur, accompanied by some wheeze in both sides of the lungs. Because of the respiratory difficulty, the patient could hardly speak. After primary physical examination, he was suspected of having foreign body airway obstruction. Around 30 minutes after admission, the patient suddenly became unconscious with HR and BP not being measured. The patient died after an hour of cardiopulmonary resuscitation.\nThis study was approved by the Academic Committee of the Institute of Forensic Science, Ministry of Justice, People's Republic of China. Written informed consents were obtained from the victim's family to publish these case details."
    ]
    
    # Run inference
    summaries = run_inference(patient_texts, model, tokenizer)

    # Print out the summaries
    for i, summary in enumerate(summaries):
        print(f"Summary {i+1}: {summary}")

