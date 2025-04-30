import sys
import os
import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
 
# app_dir = os.path.dirname(os.path.abspath(__file__)) 
# project_root = os.path.dirname(app_dir)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from src.inference import generate_summary

INFRENCE_MODEL = 'models/final_model'

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(INFRENCE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(INFRENCE_MODEL)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🩺 Clinical Note Summarizer")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "single"

tabs = {
    "🔍 Single Note": "single",
    "📁 Batch via CSV": "batch"
}
selected = st.selectbox("Choose mode", list(tabs.keys()), index=0 if st.session_state.active_tab == "single" else 1)

st.session_state.active_tab = tabs[selected]


# ---- Single NOTE OPTION----
if st.session_state.active_tab == "single":
    user_input = st.text_area("Paste clinical note here:", height=200)
    max_tokens = st.slider("Max summary length", 30, 300, 150, key="max_tokens_single")

    if st.button("Summarize", key="summarize_single") and user_input.strip():
        with st.spinner("Generating summary..."):
            summary = generate_summary(model, tokenizer, user_input, base_output_length=max_tokens)
        st.markdown("### Summary:")
        st.success(summary)
# ---- Batch NOTES OPTION ----
elif st.session_state.active_tab == "batch":
    uploaded_file = st.file_uploader("Upload CSV with a 'note' column", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Total rows: {len(df)}")
        if "note" not in df.columns:
            st.error("CSV must contain a column named 'note'")
        else:
            num_rows = st.number_input("Number of rows to summarize", min_value=1, max_value=len(df), value=min(5, len(df)), step=1)

            if st.button("Summarize Selected Rows"):
                summaries = []
                with st.spinner(f"Summarizing top {num_rows} rows..."):
                    for note in df["note"].head(num_rows):
                        if pd.isna(note) or not str(note).strip():
                            summaries.append("")
                            continue
                        summary = generate_summary(model, tokenizer, note)
                        summaries.append(summary)

                df["summary"] = ""
                df.loc[:num_rows - 1, "summary"] = summaries
                st.success(f"Summarized top {num_rows} rows.")

                csv_download = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results as CSV", data=csv_download, file_name="summarized_notes.csv", mime="text/csv")


