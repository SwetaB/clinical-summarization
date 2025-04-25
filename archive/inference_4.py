# from transformers import EncoderDecoderModel, AutoTokenizer
# import torch

# # Model setup (ClinicalBERT encoder + GPT-2 decoder)
# encoder_model = 'emilyalsentzer/Bio_ClinicalBERT'
# decoder_model = 'gpt2'

# model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)

# encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model)
# decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)
# decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

# # Example Clinical Note
# clinical_note = "Patient is a 72-year-old female admitted with dizziness, fatigue, and nausea. Examination showed elevated blood pressure and signs suggestive of possible stroke. MRI scheduled for further evaluation."

# inputs = encoder_tokenizer(clinical_note, return_tensors="pt", truncation=True, padding=True)

# model.eval()
# with torch.no_grad():
#     summary_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=50)

# summary = decoder_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print("Clinical Summary:", summary)

from transformers import EncoderDecoderModel, AutoTokenizer
import torch

# # Encoder: ClinicalBERT
# encoder_model = 'emilyalsentzer/Bio_ClinicalBERT'
# # Decoder: BART
# decoder_model = 'facebook/bart-large-cnn'

# model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)

# encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model)
# decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)

# # Example inference
# clinical_note = "Patient is a 68-year-old male admitted with severe headache, blurred vision, and elevated blood pressure. Immediate CT scan recommended."


# inputs = encoder_tokenizer(clinical_note, return_tensors="pt", truncation=True, padding=True)

# model.eval()
# with torch.no_grad():
#     summary_ids = model.generate(
#         inputs.input_ids, 
#         attention_mask=inputs.attention_mask, 
#         max_length=50
#     )

# summary = decoder_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print("Summary:", summary)

# from transformers import pipeline

# summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

# clinical_note = "Patient is a 68-year-old male admitted with severe headache, blurred vision, and elevated blood pressure. Immediate CT scan recommended."
# clinical_note =  """A 52-year-old man underwent acupuncture and cupping treatment at an illegal Chinese medicine clinic for neck and back discomfort.
# Multiple 0.25 mm × 75 mm needles were utilized and the acupuncture points were located in the middle and on both sides of the upper back 
# and the middle of the lower back. The acupuncture and subsequent cupping treatment lasted 30 minutes, respectively. The patient presented
# to the hospital with severe gasp and dyspnea about 30 hours later. Physical examinations were as follows: blood pressure (BP) 
# was 149/94 mm Hg, heart rate (HR) was 86 beats/min, and blood oxygen saturation level was 54%. The patient was lucid, was gasping, 
# and had apnea and low respiratory murmur, accompanied by some wheeze in both sides of the lungs. Because of the respiratory difficulty,
# the patient could hardly speak. After primary physical examination, he was suspected of having foreign body airway obstruction. 
# Around 30 minutes after admission, the patient suddenly became unconscious with HR and BP not being measured. 
# The patient died after an hour of cardiopulmonary resuscitation. This study was approved by the Academic 
# Committee of the Institute of Forensic Science, Ministry of Justice, People's Republic of China. Written informed 
# consents were obtained from the victim's family to publish these case details."""

# summary = summarizer(clinical_note)
# print(summary[0]['summary_text'])


# from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration

# model_name = "google/bigbird-pegasus-large-pubmed"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = BigBirdPegasusForConditionalGeneration.from_pretrained(model_name)

# # Example clinical note
# clinical_note = """A 52-year-old man underwent acupuncture and cupping treatment at an illegal Chinese medicine clinic for neck and back discomfort.
# Multiple 0.25 mm × 75 mm needles were utilized and the acupuncture points were located in the middle and on both sides of the upper back 
# and the middle of the lower back. The acupuncture and subsequent cupping treatment lasted 30 minutes, respectively. The patient presented
# to the hospital with severe gasp and dyspnea about 30 hours later. Physical examinations were as follows: blood pressure (BP) 
# was 149/94 mm Hg, heart rate (HR) was 86 beats/min, and blood oxygen saturation level was 54%. The patient was lucid, was gasping, 
# and had apnea and low respiratory murmur, accompanied by some wheeze in both sides of the lungs. Because of the respiratory difficulty,
# the patient could hardly speak. After primary physical examination, he was suspected of having foreign body airway obstruction. 
# Around 30 minutes after admission, the patient suddenly became unconscious with HR and BP not being measured. 
# The patient died after an hour of cardiopulmonary resuscitation. This study was approved by the Academic 
# Committee of the Institute of Forensic Science, Ministry of Justice, People's Republic of China. Written informed 
# consents were obtained from the victim's family to publish these case details."""

# # Tokenize input
# inputs = tokenizer(clinical_note, return_tensors="pt", truncation=True, padding="max_length", max_length=4096)

# # Generate summary
# summary_ids = model.generate(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     max_length=200,
#     num_beams=4,
#     early_stopping=True
# )

# # Decode and print summary
# summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
# print("Generated Summary:", summary)


# #######
# '''
# Generated Summary: this case highlights the need for complementary and alternative medicines ( cam ) 
# in the emergency department of a hospital.<n> cam is a branch of medicine that is not licensed by
#  the korean ministry of health and welfare.<n> cam is believed to be a form of complementary and 
#  alternative medicine ( cam ), which is closely related to acupuncture.<n> although there are 
#  regulations on the use of cam in the korean health system, there are no regulations for the use 
#  of cam in the emergency department of a hospital.<n> therefore, it is necessary to obtain a 
#  doctor's informed consent before using cam for a patient.<n> the use of cam is a form of medico - legal 
#  intervention in the korean health system.<n> the use of cam is not supported by either a prescription
# from a physician or by an insurance company.<n> therefore, the use of cam is a form of medico - legal intervention in the korean health system.'''

from rouge_score import rouge_scorer

generated_summary = """A 52-year-old man developed severe dyspnea and hypoxia approximately 30 hours after undergoing acupuncture and cupping at an unlicensed Chinese medicine clinic. Initial vitals showed BP 149/94 mmHg, HR 86 bpm, and SpO₂ at 54%. The patient was lucid but gasping with signs of apnea, wheezing, and diminished breath sounds bilaterally. Suspected of having foreign body airway obstruction, he became unresponsive 30 minutes post-admission and died despite one hour of CPR."""

gold_abstract = """Acupuncture, a component of traditional Chinese medicine, is also a well-known form of complementary and alternative medicine. Serious adverse events of acupuncture have been reported, including the acupuncture-related pneumothorax which is a rare but fatal condition sometimes. The pneumothorax was related to needle insertion in the upper back or paraspinal area and the reported victims suffered from either unilateral or bilateral pneumothorax. Postmortem computed tomography has advantages in the detection of pathologic gas and is being considered as a useful visualization tool for diagnosing the cause of death."""

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(gold_abstract, generated_summary)

print(scores)
