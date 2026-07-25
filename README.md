# English → Hindi LLM Translator

A prompt-based (few-shot) English-to-Hindi translation project using
open-source instruction-tuned LLMs via the HuggingFace Inference API.

## Why this approach

Instead of fine-tuning or training a model from scratch, this project uses
few-shot prompting with a general-purpose multilingual instruct model
(e.g. Aya-23, Llama-3.1-Instruct, Mistral-Instruct). This tests how well a
general LLM can translate purely from instructions + examples, and compares
that against a dedicated MT model (IndicTrans2) as a baseline.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxxxxxxxxxxx   # get one free at huggingface.co/settings/tokens
```

Note: some models (e.g. Llama-3.1) are gated — you must accept their license
on the HF model page before your token can call them. Aya-23 is ungated.

## Usage

**Single sentence (CLI):**
```bash
python translator.py "Where is the nearest railway station?"
```

**Batch file:**
```python
from translator import translate_file
translate_file("data/input.txt", "data/output.txt")
```

**Web demo:**
```bash
streamlit run app.py
```

**Evaluation (BLEU / chrF++):**
```bash
python evaluate.py --model CohereForAI/aya-23-8B --num_shots 5 --limit 50
```

## Project structure

```
translate_project/
├── prompts.py           # system prompt + few-shot examples
├── translator.py        # HF Inference API wrapper (single + batch)
├── evaluate.py          # BLEU/chrF++ scoring against a reference set
├── app.py               # Streamlit demo UI
├── requirements.txt
├── data/
│   └── eval_set.tsv     # sample EN-HI pairs (replace with FLORES-200/Samanantar for real eval)
└── notebook/            # for error analysis writeup
```


