"""Gradio demo for the medical Q&A QLoRA adapter — runs on Hugging Face Spaces.

Two backends, selected by env var:
  - default: ZeroGPU / GPU path — load base in 4-bit + PEFT adapter.
  - USE_GGUF=1: CPU path — run a merged q4 GGUF via llama-cpp-python (fallback
    while a ZeroGPU grant is pending). Set GGUF_REPO / GGUF_FILE accordingly.

Env vars:
  BASE_MODEL   default mistralai/Mistral-7B-Instruct-v0.3
  ADAPTER_REPO default mohitxshukla/medqa-mistral-7b-qlora   <-- set to your repo
  USE_GGUF     "1" to use the CPU GGUF backend
  GGUF_REPO    HF repo holding the GGUF
  GGUF_FILE    GGUF filename
"""

from __future__ import annotations

import os
import sys

import gradio as gr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from prompt import DISCLAIMER, build_prompt  # noqa: E402

BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "mohitxshukla/medqa-mistral-7b-qlora")
USE_GGUF = os.environ.get("USE_GGUF", "0") == "1"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
if USE_GGUF:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    _gguf_path = hf_hub_download(
        repo_id=os.environ["GGUF_REPO"],
        filename=os.environ["GGUF_FILE"],
    )
    _llm = Llama(model_path=_gguf_path, n_ctx=2048, n_threads=os.cpu_count())

    def infer(question: str) -> str:
        out = _llm(build_prompt(question), max_tokens=MAX_NEW_TOKENS, stop=["</s>"])
        return out["choices"][0]["text"].strip()

else:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=_bnb, device_map="auto", torch_dtype=torch.float16
    )
    _model = PeftModel.from_pretrained(_base, ADAPTER_REPO)
    _model.eval()
    _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    # ZeroGPU: wrap the generate call so the Space requests a GPU slice on demand.
    try:
        import spaces

        gpu_decorator = spaces.GPU(duration=60)
    except Exception:  # not on ZeroGPU — run as-is
        def gpu_decorator(fn):
            return fn

    @gpu_decorator
    @torch.no_grad()
    def infer(question: str) -> str:
        ids = _tok(build_prompt(question), return_tensors="pt").to(_model.device)
        out = _model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=_tok.pad_token_id,
        )
        return _tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def respond(message: str, history: list) -> str:
    if not message.strip():
        return "Please enter a question."
    return infer(message)


EXAMPLES = [
    "What are the early warning signs of type 2 diabetes?",
    "Is it safe to take ibuprofen and paracetamol together?",
    "What lifestyle changes help lower high blood pressure?",
    "How long is someone with the flu contagious?",
]

demo = gr.ChatInterface(
    fn=respond,
    title="🩺 MedQA Mistral-7B (QLoRA)",
    description=(
        f"Fine-tuned `{BASE_MODEL}` adapter `{ADAPTER_REPO}` on public medical Q&A.\n\n"
        f"{DISCLAIMER}"
    ),
    examples=EXAMPLES,
    type="messages",
)

if __name__ == "__main__":
    demo.launch()
