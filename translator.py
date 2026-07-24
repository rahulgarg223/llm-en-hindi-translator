"""
Translator wrapper around the HuggingFace Inference API.

Usage:
    export HF_TOKEN=hf_xxx...
    python translator.py "Where is the nearest railway station?"
Requires: pip install huggingface_hub
"""

import os
import re
import sys
import time
from typing import List, Optional

from huggingface_hub import InferenceClient

from prompts import build_chat_messages

# Any HF-hosted instruction-tuned model that supports the chat_completion
# task works here. Swap this to compare models.
DEFAULT_MODEL = "CohereForAI/aya-23-8B"

# A couple of good alternatives to try / compare against:
ALT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


class HindiTranslator:
    def __init__(self, model: str = DEFAULT_MODEL, hf_token: Optional[str] = None):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError(
                "No HuggingFace token found. Set the HF_TOKEN environment "
                "variable or pass hf_token explicitly."
            )
        self.model = model
        self.client = InferenceClient(model=model, token=token)

    def translate(self, text: str, num_shots: int = 5, max_tokens: int = 256) -> str:
        """Translate a single sentence/paragraph from English to Hindi."""
        messages = build_chat_messages(text, num_shots=num_shots)
        response = self.client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,  # deterministic output for translation
        )
        raw_output = response.choices[0].message.content
        return self._clean_output(raw_output)

    def translate_batch(
        self, texts: List[str], num_shots: int = 5, delay: float = 0.0
    ) -> List[str]:
        """
        Translate a list of sentences. `delay` adds a pause between calls
        to stay under free-tier rate limits.
        """
        results = []
        for i, text in enumerate(texts):
            try:
                results.append(self.translate(text, num_shots=num_shots))
            except Exception as e:
                print(f"[warn] failed on item {i}: {e}", file=sys.stderr)
                results.append("")
            if delay:
                time.sleep(delay)
        return results

    @staticmethod
    def _clean_output(raw: str) -> str:
        """
        Strips common artifacts models add despite instructions:
        leading 'HI:' labels, quotes, stray English preambles.
        """
        text = raw.strip()
        text = re.sub(r"^(HI|Hindi|Translation)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        text = text.strip("\"'“”")
        return text.strip()


def translate_file(input_path: str, output_path: str, model: str = DEFAULT_MODEL):
    """Reads one sentence per line from input_path, writes Hindi translations."""
    translator = HindiTranslator(model=model)
    with open(input_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    translations = translator.translate_batch(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line + "\n")

    print(f"Translated {len(lines)} lines -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python translator.py \"<English sentence>\"")
        sys.exit(1)

    text = sys.argv[1]
    translator = HindiTranslator()
    print(translator.translate(text))
