"""Before/after evaluation: base model vs QLoRA fine-tuned adapter.

Metrics on the held-out eval split:
  - Perplexity (lower = better) on the reference answers given the question.
  - ROUGE-L F1 (higher = better) of generated vs reference answer.
  - BERTScore F1 (higher = better) — semantic similarity.
Also dumps N side-by-side example generations for the model card.

Full run (GPU):
    python src/evaluate.py --base mistralai/Mistral-7B-Instruct-v0.3 \
        --adapter mohitxshukla/medqa-mistral-7b-qlora --n 200 --examples 5

CI / wiring check (CPU, tiny model, no network for data):
    python src/evaluate.py --smoke
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import SplitConfig, load_splits, smoke_dataset
from prompt import build_prompt, build_training_text

RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results_template.md"


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sequence_perplexity(model, tok, question: str, answer: str, device: str) -> float:
    """Perplexity of the answer tokens conditioned on the prompt."""
    full = build_training_text(question, answer)
    prompt = build_prompt(question)
    full_ids = tok(full, return_tensors="pt").input_ids.to(device)
    prompt_len = tok(prompt, return_tensors="pt").input_ids.shape[1]

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # only score the answer
    out = model(full_ids, labels=labels)
    return float(torch.exp(out.loss).item())


@torch.no_grad()
def generate(model, tok, question: str, device: str, max_new_tokens: int = 256) -> str:
    ids = tok(build_prompt(question), return_tensors="pt").to(device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    text = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


def text_metrics(preds: list[str], refs: list[str]) -> dict[str, float]:
    """ROUGE-L + BERTScore. Imported lazily so --smoke can skip heavy deps."""
    import evaluate as hf_evaluate

    rouge = hf_evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=preds, references=refs, rouge_types=["rougeL"])
    out = {"rougeL": float(rouge_res["rougeL"])}
    try:
        bertscore = hf_evaluate.load("bertscore")
        bs = bertscore.compute(predictions=preds, references=refs, lang="en")
        out["bertscore_f1"] = sum(bs["f1"]) / len(bs["f1"])
    except Exception as e:  # network / model download may fail offline
        print(f"[warn] bertscore skipped: {e}")
        out["bertscore_f1"] = float("nan")
    return out


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_base(base: str, smoke: bool, device: str):
    kwargs = {} if smoke else {"torch_dtype": torch.float16, "device_map": device}
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if smoke:
        model = model.to(device)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def attach_adapter(base_model, adapter: str):
    return PeftModel.from_pretrained(base_model, adapter)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def evaluate_model(model, tok, rows, device, do_text: bool) -> dict:
    ppls, preds, refs = [], [], []
    for r in rows:
        ppls.append(sequence_perplexity(model, tok, r["question"], r["answer"], device))
        if do_text:
            preds.append(generate(model, tok, r["question"], device))
            refs.append(r["answer"])
    result = {"perplexity": sum(ppls) / len(ppls)}
    if do_text and not any(math.isnan(p) for p in [result["perplexity"]]):
        result.update(text_metrics(preds, refs))
    result["_preds"] = preds
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--adapter", default=None, help="HF repo id or local path to the LoRA adapter")
    ap.add_argument("--n", type=int, default=200, help="eval rows for metrics")
    ap.add_argument("--examples", type=int, default=5, help="side-by-side examples to dump")
    ap.add_argument("--smoke", action="store_true", help="tiny model + tiny data, CPU, CI wiring check")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        base_id = "hf-internal-testing/tiny-random-MistralForCausalLM"
        rows = list(smoke_dataset(4))
        do_text = False  # skip heavy ROUGE/BERTScore downloads in CI
        n = len(rows)
    else:
        base_id = args.base
        ds = load_splits(SplitConfig())
        rows = list(ds["eval"])[: args.n]
        do_text = True
        n = args.n

    print(f"Loading base: {base_id}")
    base_model, tok = load_base(base_id, args.smoke, device)
    base_res = evaluate_model(base_model, tok, rows, device, do_text)
    print(f"[base] {({k: v for k, v in base_res.items() if not k.startswith('_')})}")

    ft_res = None
    if args.adapter:
        print(f"Loading adapter: {args.adapter}")
        ft_model = attach_adapter(base_model, args.adapter)
        ft_res = evaluate_model(ft_model, tok, rows, device, do_text)
        print(f"[fine-tuned] {({k: v for k, v in ft_res.items() if not k.startswith('_')})}")

    if not args.smoke:
        write_results(args, n, base_res, ft_res, rows)
    else:
        print("\nSmoke eval OK — pipeline wired correctly.")


def write_results(args, n, base_res, ft_res, rows) -> None:
    def fmt(res, key):
        return f"{res[key]:.4f}" if res and key in res else "—"

    lines = [
        "# Evaluation results",
        "",
        f"- Base model: `{args.base}`",
        f"- Adapter: `{args.adapter}`",
        f"- Held-out eval rows: {n}",
        "",
        "| Metric | Base | Fine-tuned | Better |",
        "|---|---|---|---|",
        f"| Perplexity ↓ | {fmt(base_res, 'perplexity')} | {fmt(ft_res, 'perplexity')} | lower |",
        f"| ROUGE-L ↑ | {fmt(base_res, 'rougeL')} | {fmt(ft_res, 'rougeL')} | higher |",
        f"| BERTScore F1 ↑ | {fmt(base_res, 'bertscore_f1')} | {fmt(ft_res, 'bertscore_f1')} | higher |",
        "",
        "## Example generations",
        "",
    ]
    for i in range(min(args.examples, len(rows))):
        lines += [
            f"### Example {i + 1}",
            f"**Question:** {rows[i]['question']}",
            "",
            f"**Reference:** {rows[i]['answer']}",
            "",
        ]
        if base_res.get("_preds"):
            lines += [f"**Base:** {base_res['_preds'][i]}", ""]
        if ft_res and ft_res.get("_preds"):
            lines += [f"**Fine-tuned:** {ft_res['_preds'][i]}", ""]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
