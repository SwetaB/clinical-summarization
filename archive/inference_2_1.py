import torch
# *** Changed from Bart to T5 ***
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import sys

def load_model_and_tokenizer(model_path: str):
    """
    Load the model and tokenizer for inference.
    """
    print(f"Attempting to load model and tokenizer from: {model_path}")
    try:
        # *** Changed from Bart to T5 ***
        # model = T5ForConditionalGeneration.from_pretrained(model_path)
        # tokenizer = T5Tokenizer.from_pretrained(model_path)
        # model = BartForConditionalGeneration.from_pretrained(model_path)
        # tokenizer = BartTokenizer.from_pretrained(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        # The error message from the library is quite informative here, keep it
        raise OSError(f"Error loading model or tokenizer from {model_path}. Details: {e}")

def generate_summary(text: str, model, tokenizer, max_length: int = 150, num_beams: int = 4) -> str:
    """
    Generate a summary for the input text.
    """
    # T5 models often use a prefix like "summarize: " to indicate the task
    input_text = "summarize: " + text

    # T5 tokenizer uses 'return_tensors="pt"' for PyTorch
    # max_length and truncation handle input size
    # padding=True pads the input sequence
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True) # T5 models often have a smaller max input length like 512

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"Using device: {device}")

    # max_length here is for the generated output summary length
    summary_ids = model.generate(inputs['input_ids'], max_length=max_length, num_beams=num_beams, early_stopping=True)

    # Decode the summary back to text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # T5 output usually doesn't have the "summarize: " prefix, but strip whitespace just in case.
    summary = summary.strip()

    return summary

def run_inference(input_texts: list[str], model, tokenizer) -> list[str]:
    """
    Perform inference (summarization) for multiple input texts.
    """
    summaries = []
    if not input_texts:
        print("No input texts provided.")
        return []

    print(f"\nStarting summarization for {len(input_texts)} text(s)...")
    for i, text in enumerate(input_texts):
        print(f"\nProcessing text {i+1}/{len(input_texts)}...")
        if not isinstance(text, str) or not text.strip():
            print(f"Skipping text {i+1}: Invalid or empty input.")
            summaries.append(f"Error: Invalid or empty input text for item {i+1}")
            continue

        try:
            # Warn if text is significantly longer than typical T5 input max_length (e.g., 512 tokens)
            if len(tokenizer.encode(text)) > 500: # Check token count, not char count
                 print(f"Warning: Text {i+1} is very long ({len(tokenizer.encode(text))} tokens). It will be truncated by the tokenizer.")

            summary = generate_summary(text, model, tokenizer)
            summaries.append(summary)
            print(f"Finished processing text {i+1}.")
        except Exception as e:
            print(f"Error processing text {i+1}: {e}")
            summaries.append(f"Error summarizing text {i+1}: {e}")
    print("\nSummarization complete.")
    return summaries

if __name__ == "__main__":
    # Set the model path to a clinically relevant summarization model on Hugging Face Hub.
    # 'Falconsai/medical_summarization' is a T5-base model fine-tuned on medical data.
    # MODEL_PATH = 'Falconsai/medical_summarization'
    # MODEL_PATH = 'google/flan-t5-base'
    # MODEL_PATH = 'google/pegasus-xsum'
    # MODEL_PATH = 'facebook/bart-large-cnn'
    MODEL_PATH = "emilyalsentzer/Bio_ClinicalBERT"

    try:
        # Load the model and tokenizer (will download if not cached)
        model, tokenizer = load_model_and_tokenizer(model_path=MODEL_PATH)
    except OSError as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Sample clinical texts. Replace with your actual data.
    patient_texts = [
        "A 52-year-old man underwent acupuncture and cupping treatment at an illegal Chinese medicine clinic for neck and back discomfort. Multiple 0.25 mm × 75 mm needles were utilized and the acupuncture points were located in the middle and on both sides of the upper back and the middle of the lower back. The acupuncture and subsequent cupping treatment lasted 30 minutes, respectively. The patient presented to the hospital with severe gasp and dyspnea about 30 hours later. Physical examinations were as follows: blood pressure (BP) was 149/94 mm Hg, heart rate (HR) was 86 beats/min, and blood oxygen saturation level was 54%. The patient was lucid, was gasping, and had apnea and low respiratory murmur, accompanied by some wheeze in both sides of the lungs. Because of the respiratory difficulty, the patient could hardly speak. After primary physical examination, he was suspected of having foreign body airway obstruction. Around 30 minutes after admission, the patient suddenly became unconscious with HR and BP not being measured. The patient died after an hour of cardiopulmonary resuscitation. This study was approved by the Academic Committee of the Institute of Forensic Science, Ministry of Justice, People's Republic of China. Written informed consents were obtained from the victim's family to publish these case details.",
        "A 35-year-old man without systemic disease first attended our clinic in August 2004 for bilateral JOAG. He denied having a family history of glaucoma, but his uncle had been diagnosed with LHON. When he was undergoing therapy with timolol 0.5%, his IOP was approximately 20 mmHg in both eyes. His BCVA gradually decreased from 20/200 in both eyes in 2006 to counting fingers at 25-30 cm in both eyes in 2016. Gonioscopy revealed a normal iridocorneal angle; pachymetric measurements were 561 μm in the right eye and 563 μm in the left eye. Fundoscopic examination revealed paled optic disc with enlarged disc cupping of the optic nerves with sectorial excavation and reduction of the neural rim in both eyes (Fig. ). OCTA disclosed diffuse RNFL thinning and a decreased peripapillary vascularity in both eyes (Fig. ). The VF (30–2 SITA standard) was characterized by progressive central scotoma in both eyes. The ERG was subnormal in both eyes, and the pattern ERG revealed decreased N95 amplitudes in both eyes (Fig. ). The genetic test revealed an ND4 m11778G > A mtDNA mutation, which is pathognomonic for LHON."
        # "Patient is a 68-year-old male admitted with acute exacerbation of chronic obstructive pulmonary disease (COPD). Presented to the emergency department with increased shortness of breath, cough with yellow sputum, and wheezing for the past three days. History includes smoking 1 pack/day for 40 years, hypertension, and coronary artery disease. Medications: Albuterol inhaler PRN, Symbicort BID, Lisinopril 10mg daily. Physical exam: tachycardic (HR 110), tachypneic (RR 24), decreased breath sounds bilaterally with expiratory wheezing. SaO2 88% on room air. Chest X-ray shows hyperinflation but no consolidation. ABGs: pH 7.32, pCO2 65, pO2 55. Started on IV steroids, antibiotics, and supplemental oxygen via nasal cannula to maintain SaO2 > 90%.",
        # "Consultation Note: Patient seen for follow-up regarding uncontrolled type 2 diabetes mellitus. A1C is 9.8% (up from 8.5% three months ago). Patient reports poor adherence to diet and exercise recommendations. Currently on Metformin 1000mg BID and Glipizide 5mg daily. Denies symptoms of hypoglycemia. Discussed importance of lifestyle changes and medication adherence. Increased Glipizide to 10mg daily. Scheduled return visit in 6 weeks.",
        # "Brief note: Patient tolerated procedure well. No complications noted.",
        # "A 34-year-old Chinese woman, gravida 1, para 0, was referred to our hospital for thickened nuchal translucency (3.4 mm) in one fetus of dichorionic diamniotic twins at 13 weeks of pregnancy. She underwent in vitro fertilization and embryo transfer (IVF-ET) because her husband was oligoasthenospermia. Two embryos were transferred to the uterus. Transvaginal ultrasound revealed an unremarkable dichorionic twin pregnancy in the first trimester. Noninvasive prenatal testing (NIPT) was performed at 15 weeks of gestation, showing a low-risk for fetal 21, 13 and 18 trisomy. After informed consent was obtained, she underwent amniocentesis for further molecular analysis at 17 weeks of gestation. The results of CMA showed a gain of the entire short arm of chromosome 12 in approximately 80% of cells in the fetus with thickened nuchal translucency, while normal in the other fetus (Figure E). SNP array analysis confirmed that the twins were dizygotic. Then, a second amniocentesis was offered to confirm the tetrasomy using FISH and G-banding karyotyping at 20 weeks of gestation. The karyotyping showed that the abnormal fetus was 47,XX,i(12p)[40]/46,XX[10] (Figure A). FISH analysis confirmed tetrasomy 12p in 80% (20/25) of cells (Figure B). Both the karyotype and FISH results of the other fetus showed a normal female (Figure C,D).\nThe decision to terminate the abnormal fetus was difficult for the parents due to the wide spectrum of PKS manifestations. At 20 weeks of gestation, more abnormalities, including severely shortened humerus and femur (<−6 SD) and mild lateral ventriculomegaly, were revealed by three-dimensional ultrasound. Meanwhile, head circumference, abdominal circumference, and biparietal diameter were in the normal ranges. In consideration of the ultrasonic and cytogenetic findings, the parents opted for selective termination. Transabdominal intrathoracic injection of potassium chloride (KCl) into the heart of the fetus with 12p tetrasomy was performed successfully at 23 weeks of gestation. Before the injection of KCl, heart blood was obtained and the karyotype analysis of heart blood in the abnormal fetus showed 47,XX,i(12p)[12]/46,XX[23]. Subsequently, ultrasounds were performed regularly, and the remaining fetus showed normal biometric parameters. A healthy female baby was born by normal vaginal delivery at term"
    ]

    summaries = run_inference(patient_texts, model, tokenizer)

    print("\n" + "="*30)
    print("--- GENERATED SUMMARIES ---")
    print("="*30 + "\n")

    for i, summary in enumerate(summaries):
        print(f"--- Summary {i+1} ---")
        print(summary)
        print("-" * 20)

    print("\n" + "="*30)
    print("--- END OF SUMMARIES ---")
    print("="*30 + "\n")



'''
MODEL_PATH = 'Falconsai/medical_summarization'
==============================
--- GENERATED SUMMARIES ---
==============================

--- Summary 1 ---
a 68-year-old male with acute exacerbation of chronic obstructive pulmonary disease (COPD) is admitted with acute exacerbation of chronic obstructive pulmonary disease ( COPD ). presenting to the emergency department with increased shortness of breath, cough with yellow sputum, and wheezing for the past three days. history includes smoking 1 pack/day for 40 years, hypertension, and coronary artery disease.
--------------------
--- Summary 2 ---
type 2 diabetes mellitus ( a1C) is 9.8% (up from 8.5% three months ago). Patient reports poor adherence to diet and exercise recommendations. Currently on Metformin 1000mg BID and Glipizide 5mg daily. Denies symptoms of hypoglycemia. Discussed importance of lifestyle changes and medication adherence. Increased Glipizide to 10mg daily. Scheduled return visit in 6 weeks.
--------------------
--- Summary 3 ---
patient tolerated procedure well. no complications noted noted. no complications noted. patient tolerated procedure well.
'''


'''
MODEL_PATH = 'google/flan-t5-base'
==============================
--- GENERATED SUMMARIES ---
==============================

--- Summary 1 ---
Acute exacerbation of chronic obstructive pulmonary disease.
--------------------
--- Summary 2 ---
A1C is 9.8% (up from 8.5% three months ago)
--------------------
--- Summary 3 ---
The patient was able to perform the procedure without any complications.
--------------------

==============================
--- END OF SUMMARIES ---
==============================
'''

'''
# Bullshit - MODEL_PATH = 'google/pegasus-xsum'
==============================
--- GENERATED SUMMARIES ---
==============================

--- Summary 1 ---
gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip
--------------------
--- Summary 2 ---
gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting busting
--------------------
--- Summary 3 ---
gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossip gossipMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMicheleMichele
--------------------

==============================
--- END OF SUMMARIES ---
==============================

'''