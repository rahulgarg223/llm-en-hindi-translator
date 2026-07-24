"""
Evaluation script: computes BLEU and chrF++ scores for the translator
against a reference parallel dataset (e.g. a FLORES-200 or Samanantar
subset in data/eval_set.tsv).

Usage:
    export HF_TOKEN=hf_xxx...
    python evaluate.py --model CohereForAI/aya-23-8B --num_shots 5 --limit 50

    
Requires: pip install sacrebleu
"""

import argparse
import csv

import sacrebleu

from translator import HindiTranslator, DEFAULT_MODEL


def load_eval_set(path: str, limit: int = None):
    pairs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            en, hi = row[0].strip(), row[1].strip()
            if en and hi:
                pairs.append((en, hi))
    if limit:
        pairs = pairs[:limit]
    return pairs


def run_eval(data_path: str, model: str, num_shots: int, limit: int = None):
    pairs = load_eval_set(data_path, limit=limit)
    sources = [p[0] for p in pairs]
    references = [p[1] for p in pairs]

    translator = HindiTranslator(model=model)
    hypotheses = translator.translate_batch(sources, num_shots=num_shots)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)  # chrF++

    print(f"Model: {model}")
    print(f"Few-shot examples used: {num_shots}")
    print(f"Sentences evaluated: {len(pairs)}")
    print(f"BLEU:   {bleu.score:.2f}")
    print(f"chrF++: {chrf.score:.2f}")

    # Save per-sentence outputs for error analysis
    with open("data/eval_outputs.tsv", "w", encoding="utf-8") as f:
        f.write("source\treference\thypothesis\n")
        for src, ref, hyp in zip(sources, references, hypotheses):
            f.write(f"{src}\t{ref}\t{hyp}\n")
    print("Per-sentence outputs saved to data/eval_outputs.tsv for error analysis.")

    return {"bleu": bleu.score, "chrf": chrf.score}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eval_set.tsv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num_shots", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_eval(args.data, args.model, args.num_shots, args.limit)
