"""
Streamlit demo app for the English -> Hindi LLM translator.

Run with:
    export HF_TOKEN=hf_xxx...
    streamlit run app.py
"""

import os

import streamlit as st

from translator import HindiTranslator, DEFAULT_MODEL, ALT_MODELS

st.set_page_config(page_title="EN → HI Translator", page_icon="🌐")

st.title("🌐 English → Hindi Translator")
st.caption("LLM-powered translation using few-shot prompting (HuggingFace Inference API)")

with st.sidebar:
    st.header("Settings")
    model_options = [DEFAULT_MODEL] + ALT_MODELS
    model = st.selectbox("Model", model_options, index=0)
    num_shots = st.slider("Few-shot examples", min_value=0, max_value=5, value=5)
    hf_token_input = st.text_input(
        "HF Token (optional if HF_TOKEN env var is set)", type="password"
    )
    st.markdown("---")
    st.markdown(
        "Get a free HuggingFace token at "
        "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)"
    )

st.subheader("Single sentence translation")
source_text = st.text_area("English text", height=120, placeholder="Type a sentence to translate...")

if st.button("Translate", type="primary"):
    if not source_text.strip():
        st.warning("Please enter some text.")
    else:
        token = hf_token_input or os.environ.get("HF_TOKEN")
        if not token:
            st.error("No HF token provided. Set HF_TOKEN env var or paste it in the sidebar.")
        else:
            with st.spinner("Translating..."):
                try:
                    translator = HindiTranslator(model=model, hf_token=token)
                    result = translator.translate(source_text, num_shots=num_shots)
                    st.success("Translation complete")
                    st.text_area("Hindi translation", value=result, height=120)
                except Exception as e:
                    st.error(f"Translation failed: {e}")

st.markdown("---")
st.subheader("Batch translation (upload a .txt file, one sentence per line)")
uploaded_file = st.file_uploader("Upload file", type=["txt"])

if uploaded_file is not None and st.button("Translate file"):
    token = hf_token_input or os.environ.get("HF_TOKEN")
    if not token:
        st.error("No HF token provided.")
    else:
        lines = [
            line.decode("utf-8").strip()
            for line in uploaded_file.readlines()
            if line.strip()
        ]
        translator = HindiTranslator(model=model, hf_token=token)
        progress = st.progress(0)
        results = []
        for i, line in enumerate(lines):
            try:
                results.append(translator.translate(line, num_shots=num_shots))
            except Exception as e:
                results.append(f"[error: {e}]")
            progress.progress((i + 1) / len(lines))

        output_text = "\n".join(results)
        st.text_area("Translated output", value=output_text, height=300)
        st.download_button(
            "Download translations (.txt)",
            data=output_text,
            file_name="translated_output.txt",
            mime="text/plain",
        )
