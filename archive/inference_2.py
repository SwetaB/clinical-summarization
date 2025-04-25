# Using the pre-trained facebook/bart-large-cnn model
from transformers import BartForConditionalGeneration, BartTokenizer

def generate_summary(text, model, tokenizer, max_length=150, min_length=40, num_beams=5, temperature=1.0, top_k=50):
    """
    Generate a summary for the input text using the fine-tuned model.

    Args:
        text (str): The input text to summarize.
        model: The fine-tuned model.
        tokenizer: The tokenizer for the model.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.
        num_beams (int): Number of beams for beam search during generation.
        temperature (float): The value used to module the next token probabilities.
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.

    Returns:
        str: The generated summary.
    """
    # Add the "summarize: " prefix
    input_text = "summarize: " + text

    # Tokenize the input text using the tokenizer
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

    # Generate the summary using the model
    summary_ids = model.generate(inputs['input_ids'],
                                 max_length=max_length,
                                 min_length=min_length, # Added min_length
                                 num_beams=num_beams,
                                 early_stopping=True,
                                 length_penalty=2.0 # Added length penalty to encourage longer summaries within limits
                                 )

    # Decode the summary back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Remove the "summarize: " prefix from the generated summary if it appears
    summary = summary.replace("summarize: ", "").strip()

    return summary

# Load the pre-trained BART model and tokenizer
# This step might take a few moments as it downloads the model weights if not cached.
# 'facebook/bart-large-cnn'
print("Loading tokenizer...")
tokenizer = BartTokenizer.from_pretrained('microsoft/BioBART')
print("Loading model...")
model = BartForConditionalGeneration.from_pretrained('microsoft/BioBART')
print("Model and tokenizer loaded.")

# --- Longer and More Detailed Test Input ---

# Sample clinical texts. Replace with your actual data.
patient_text = """
Patient is a 72-year-old male with a significant past medical history including coronary artery disease status post coronary artery bypass grafting (CABG x3 vessels) approximately 10 years ago, type 2 diabetes mellitus managed with metformin and basal insulin, chronic kidney disease stage 3 (baseline creatinine around 1.8 mg/dL), hypertension, hyperlipidemia, and mild chronic obstructive pulmonary disease (COPD).

He presented to the Emergency Department complaining of progressive shortness of breath over the past 3 days, which has now worsened to occurring even at rest. He reports needing to sleep upright on three pillows (orthopnea) and waking up suddenly at night gasping for air (paroxysmal nocturnal dyspnea - PND). He also notes significantly increased swelling in both lower legs, described as 3+ pitting edema extending up to the mid-shins. He denies current chest pain, fever, or productive cough, although he mentions having a mild, non-productive cough the previous week. He reports decreased urine output over the last 48 hours. Regarding compliance, he admits he might have missed a couple of doses of his diuretic (furosemide) this week and recalls eating a particularly salty meal at a restaurant 4 days prior to presentation.

Review of Systems: Positive for generalized fatigue and audible wheezing. Negative for palpitations, syncope, nausea, vomiting, or abdominal pain.

Physical Examination: Vital signs on arrival showed Blood Pressure 165/95 mmHg, Heart Rate 105 bpm (regular), Respiratory Rate 24 breaths/min, Oxygen Saturation 91% on room air, Temperature 37.2 C (98.9 F). General appearance revealed an uncomfortable male in moderate respiratory distress. Neck examination showed jugular venous distension (JVD) elevated to the angle of the jaw. Pulmonary exam revealed crackles auscultated halfway up bilaterally, with scattered expiratory wheezes. Cardiovascular exam showed a tachycardic rate with a regular rhythm and a notable S3 gallop; no murmurs appreciated. Abdominal exam was soft, non-tender, with positive hepatojugular reflux. Extremities examination confirmed 3+ bilateral pitting edema extending to the mid-shins.

Initial Diagnostic Workup: Electrocardiogram (ECG) demonstrated sinus tachycardia at 105 bpm, findings consistent with left ventricular hypertrophy (LVH), but no acute ischemic ST-segment or T-wave changes. Laboratory results were significant for a markedly elevated B-type Natriuretic Peptide (BNP) level of 2500 pg/mL. Basic metabolic panel showed an elevated creatinine of 2.1 mg/dL (up from his baseline of 1.8) and mild hyponatremia. Troponin I level was negative on initial draw. A portable chest X-ray revealed cardiomegaly, significant pulmonary vascular congestion, interstitial edema, and small bilateral pleural effusions, consistent with pulmonary edema.

Assessment and Plan: The primary assessment is acute decompensated heart failure (ADHF), likely precipitated by a combination of medication (diuretic) non-adherence and recent dietary indiscretion (high salt intake), superimposed on his known chronic systolic dysfunction, chronic kidney disease, and hypertension. Differential diagnoses include pulmonary embolism (less likely given negative troponin and lack of pleuritic chest pain) or pneumonia (less likely given lack of fever/productive cough). The plan includes admission to a telemetry monitored bed for close cardiac monitoring. Initiate intravenous diuresis with Furosemide 80mg IV push. Provide supplemental oxygen via nasal cannula titrated to maintain SpO2 greater than 94%. Implement strict intake and output monitoring, daily weights, and a fluid-restricted, low-sodium diet. Consult Cardiology service for further management recommendations and possible optimization of guideline-directed medical therapy. Continue home medications, adjusting insulin regimen based on point-of-care glucose monitoring. Monitor electrolytes and renal function closely due to diuresis and underlying CKD.
"""

# Generate summary
print("Generating summary...")
# Adjusting generation parameters slightly for potentially better results with longer text
summary = generate_summary(patient_text, model, tokenizer, max_length=512, min_length=75, num_beams=6, temperature=1.0, top_k=50)
print(f"Summary: {summary}")

# # Using the pre-trained facebook/bart-large-cnn model
# from transformers import BartForConditionalGeneration, BartTokenizer

# def generate_summary(text, model, tokenizer, max_length=150, num_beams=5, temperature=1.0, top_k=50):
#     """
#     Generate a summary for the input text using the fine-tuned model.
    
#     Args:
#         text (str): The input text to summarize.
#         model: The fine-tuned model.
#         tokenizer: The tokenizer for the model.
#         max_length (int): Maximum length of the summary.
#         num_beams (int): Number of beams for beam search during generation.
        
#     Returns:
#         str: The generated summary.
#     """
#     # Add the "summarize: " prefix
#     input_text = "summarize: " + text

#     # Tokenize the input text using the tokenizer
#     inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length")

#     # Generate the summary using the model
#     summary_ids = model.generate(inputs['input_ids'], 
#                                  max_length=max_length,
#                                  num_beams=num_beams,
#                                  early_stopping=True,
#                                  temperature=temperature, 
#                                  top_k=top_k)

#     # Decode the summary back to text
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
#     # Remove the "summarize: " prefix from the generated summary if it appears
#     summary = summary.replace("summarize: ", "").strip()
    
#     return summary

# # Load the pre-trained BART model and tokenizer
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
# model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# # Test input
# patient_text = "60-year-old female with hypertension, diabetes, and hyperlipidemia, presenting with shortness of breath, fatigue, and chest pain. Elevated blood pressure, heart rate, and mild ST elevation on ECG. Suspected acute coronary syndrome."

# # Generate summary
# summary = generate_summary(patient_text, model, tokenizer)
# print(f"Summary: {summary}")
