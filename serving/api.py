"""FastAPI serving endpoint for the medical Q&A adapter.

Loads base + PEFT adapter once at startup, exposes:
  GET  /health    -> {"status": "ok", "model": ...}
  POST /generate  -> {"answer": ...}

Run locally:
    uvicorn serving.api:app --host 0.0.0.0 --port 8000
Or via Docker:
    docker compose -f serving/docker-compose.yml up
"""

from __future__ import annotations

import os
import sys

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from prompt import DISCLAIMER, build_prompt  # noqa: E402

BASE_MODEL = os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "mohitxshukla/medqa-mistral-7b-qlora")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
LOAD_4BIT = os.environ.get("LOAD_4BIT", "1") == "1"  # set 0 for CPU-only boxes

app = FastAPI(title="MedQA Mistral-7B QLoRA API", version="1.0.0")

_model = None
_tok = None


def _load():
    global _model, _tok
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if LOAD_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **kwargs)
    _model = PeftModel.from_pretrained(base, ADAPTER_REPO).eval()
    _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token


@app.on_event("startup")
def startup() -> None:
    _load()


class GenRequest(BaseModel):
    question: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS, ge=1, le=1024)


class GenResponse(BaseModel):
    answer: str
    disclaimer: str = DISCLAIMER


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": f"{BASE_MODEL} + {ADAPTER_REPO}", "loaded": _model is not None}


@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest) -> GenResponse:
    import torch

    ids = _tok(build_prompt(req.question), return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.generate(
            **ids,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=_tok.pad_token_id,
        )
    answer = _tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return GenResponse(answer=answer)
