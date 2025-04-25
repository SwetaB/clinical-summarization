import openai
import os
import pandas as pd
import time
import random
from datetime import datetime
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Prompt template for structured summary
def create_structured_prompt(case_text):
    return (
        "Summarize the following clinical case using the structured format:\n\n"
        "Format:\n"
        "A [age]-year-old [sex] presented with [symptom]. Intervention included [treatment]. Outcome was [result].\n\n"
        f"Clinical Case:\n{case_text}\n\nStructured Summary:"
    )

# Check if structured summary looks valid
def is_structured_summary_good(summary):
    required_phrases = ["A ", "-year-old", "presented with", "Intervention included", "Outcome was"]
    return all(phrase in summary for phrase in required_phrases)

# Safe call with retry logic
def safe_call_gpt(prompt, model="gpt-3.5-turbo", max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.1,
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
            wait_time = random.uniform(5, 15)
            print(f" Rate limit hit. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except openai.APIError as e:
            wait_time = random.uniform(5, 15)
            print(f" API Error: {e}. Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except Exception as e:
            print(f" Unexpected error: {e}")
            raise e

    raise Exception(f" Failed after {max_retries} retries.")

# Main function
def generate_structured_summaries(cases, checkpoint_every=500):
    results = []
    total_tokens = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = f"./data/summaries/structured_summaries_{timestamp}"

    for idx, case_report in enumerate(cases):
        if not case_report.strip():
            continue

        prompt = create_structured_prompt(case_report)

        # Try with GPT-3.5 first
        summary, usage = safe_call_gpt(prompt, model="gpt-3.5-turbo")
        total_tokens += usage["total_tokens"]

        # Validate
        if not is_structured_summary_good(summary):
            print(f"Bad structure detected for case {idx+1}. Retrying with GPT-4...")
            summary, usage = safe_call_gpt(prompt, model="gpt-4")
            total_tokens += usage["total_tokens"]

        results.append({
            "case": case_report,
            "structured_summary": summary
        })

        # Every checkpoint_every cases, save progress
        if (idx + 1) % checkpoint_every == 0:
            temp_df = pd.DataFrame(results)
            checkpoint_filename = f"{output_prefix}_checkpoint_{idx+1}.csv"
            temp_df.to_csv(checkpoint_filename, index=False)
            print(f"Checkpoint saved at {checkpoint_filename} after {idx+1} cases.")

    # Save final full results
    final_df = pd.DataFrame(results)
    final_filename = f"{output_prefix}_final.csv"
    final_df.to_csv(final_filename, index=False)

    return final_filename, total_tokens

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

    final_filename, total_tokens = generate_structured_summaries(clinical_cases)

    cost_estimate = (total_tokens / 1000) * 0.09  # assuming worst case GPT-4 cost

    print(f"\n Final summaries saved to '{final_filename}'")
    print(f" Total tokens used: {total_tokens}")
    print(f" Estimated cost: ${cost_estimate:.2f}")




