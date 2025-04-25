import os
import pandas as pd
from datetime import datetime
import time
import random
from openai import OpenAI
import openai

from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set your OpenAI API key

# Prompt templates
def create_structured_prompt(case_text):
    return (
        "Summarize the following clinical case using the structured format:\n\n"
        "Format:\n"
        "A [age]-year-old [sex] presented with [symptom]. Intervention included [treatment]. Outcome was [result].\n\n"
        f"Clinical Case:\n{case_text}\n\nStructured Summary:"
    )

def create_freeform_prompt(case_text):
    return (
        "Summarize the following clinical case in 3–4 sentences using a scientific, PubMed-style tone. "
        "Focus on symptoms, interventions, and outcomes.\n\n"
        f"Clinical Case:\n{case_text}\n\nSummary:"
    )

# GPT query with token logging
def call_gpt(prompt, model="gpt-4", top_p=0.2, temperature=0.3, max_retries=5):
    

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=300,
            )
            summary = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return summary, usage

        except openai.RateLimitError:
            wait_time = random.uniform(5, 15)  # random wait between 5-15 seconds
            print(f"Rate limit hit. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except openai.APIError as e:
            wait_time = random.uniform(5, 15)
            print(f"API Error: {e}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise e  # re-raise if something else (like bad input)

    raise Exception(f"Failed after {max_retries} retries.")

# Main function
def generate_summaries(case_list):
    all_outputs = []
    total_tokens = 0
    for idx, case in enumerate(case_list):
        if not case.strip():
            continue
        try:
            structured_summary, usage_structured = call_gpt(create_structured_prompt(case), top_p=0.1, temperature=0.2)
            # freeform_summary, usage_freeform = call_gpt(create_freeform_prompt(case), temperature=0.3)

            all_outputs.append({
                "case": case,
                "structured_summary": structured_summary
                # "freeform_summary": freeform_summary
            })
            # total_tokens += usage_structured["total_tokens"] + usage_freeform["total_tokens"]
            total_tokens += usage_structured["total_tokens"]

            print(f"Processed case {idx+1}/{len(case_list)} — Tokens so far: {total_tokens}")
        except Exception as e:
            print(f"Error in case {idx+1}: {e}")
    return pd.DataFrame(all_outputs), total_tokens

# Example usage
if __name__ == "__main__":
    clinical_cases = [
        "A 39-year-old man was hospitalized due to an increasingly reduced general health condition, after persistent fever and dry cough for 2 weeks...",
        "One week after a positive COVID-19 result, this 57-year-old male was admitted to the ICU because of oxygen desaturation (70%)...",
        """A 39-year-old man was hospitalized due to an increasingly reduced general health condition, after 
        persistent fever and dry cough for 2 weeks. The patient initially needed 4 L/min of oxygen, had a rapid and 
        shallow breathing pattern at rest and became severely breathless during minor physical activities. In the beginning,
        physical therapy focused on patient education about dyspnea-relieving positions, the importance of regular mobilization, 
        and deep-breathing exercises. However, it quickly became evident that his anxiety from fear of dying and worries 
        about his future aggravated his dyspnea and vice versa. The patient was so dyspneic, anxious, and weak that he 
        was barely able to walk to the toilet. To counter this vicious circle, the physical therapist actively listened to the 
        patient, explained why he was experiencing breathlessness, and tested suitable positions to relieve his dyspnea. 
        He seemed to benefit from the education and the relaxing breathing exercises, as seen on day 2, when his r
        espiratory rate could be reduced from 30 breaths/min to 22 breaths/min and his oxygen saturation increased 
        from 92% to 96% on 4 L/min oxygen after guiding him through some deep-breathing exercises. Over the next 
        days, his dyspnea and anxiety started to alleviate and he regained his self-confidence. Therapy was 
        progressively shifted to walking and strength training and the patient rapidly advanced to walk 350 m 
        without a walking aid or supplemental oxygen before his discharge home"""
    ]

    df, total_tokens = generate_summaries(clinical_cases)

    # Estimate cost (gpt-4 pricing ~$0.03 per 1k prompt tokens and ~$0.06 per 1k completion tokens)
    cost_estimate = (total_tokens / 1000) * 0.09  # rough average

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"./data/summaries/gpt3.5-turbo_summaries_{timestamp}.csv"

    df.to_csv(output_filename, index=False)

    print(f"\n Summaries saved to '{output_filename}'")
    print(f"Total tokens used: {total_tokens}")
    print(f"Estimated cost: ${cost_estimate:.4f}")