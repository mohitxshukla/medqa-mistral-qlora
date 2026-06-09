# MedQA Mistral-7B (QLoRA) — end-to-end portfolio project

Fine-tune `Mistral-7B-Instruct-v0.3` on **public medical Q&A** with **QLoRA**,
publish the adapter + model card to Hugging Face, and ship a live **Gradio Space**
demo — wrapped with a **FastAPI** serving endpoint, **Docker**, and **GitHub
Actions** CI.

> ⚠️ Educational information only — not medical advice.

## What's here

| Path | Purpose |
|---|---|
| `notebooks/train_qlora_colab.ipynb` | One-click QLoRA training on a free Colab T4 → pushes adapter to the Hub |
| `src/prompt.py` | Shared Mistral chat formatting (train = eval = serve) |
| `src/data.py` | Load + format + dedup `lavita/ChatDoctor-HealthCareMagic-100k` |
| `src/train.py` | CLI mirror of the notebook (QLoRA via TRL `SFTTrainer`) |
| `src/evaluate.py` | Before/after: perplexity, ROUGE-L, BERTScore + example dump |
| `model_card/README.md` | Hugging Face model repo card |
| `app/` | Gradio Space (ZeroGPU, with CPU GGUF fallback) |
| `serving/` | FastAPI `/generate` + Dockerfile + compose |
| `.github/workflows/eval.yml` | Lint + smoke-eval CI |

## Pipeline

```
ChatDoctor data ──▶ format (prompt.py) ──▶ QLoRA train (Colab T4) ──▶ adapter on HF Hub
                                                   │
                          evaluate.py (base vs FT) ┘──▶ metrics ──▶ model card
                                                   │
                            app/ (Gradio Space)  ◀──┴──▶  serving/ (FastAPI + Docker)
```

## Quick start

### 1. Local wiring check (no GPU, no training)
```bash
pip install ruff torch==2.4.0 transformers==4.46.2 peft==0.13.2 datasets==3.1.0 accelerate==1.1.1
ruff check src serving app
python src/evaluate.py --smoke      # tiny model, proves the eval pipeline
```

### 2. Train (free Colab T4, ~2–4h) — **you run this**
1. Accept the license at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
2. Open `notebooks/train_qlora_colab.ipynb` in Colab → Runtime = **T4 GPU** → Run all.
   It logs in to HF, trains, and pushes to `mohitxshukla/medqa-mistral-7b-qlora`.

### 3. Evaluate + fill the model card — **you run this**
```bash
python src/evaluate.py --base mistralai/Mistral-7B-Instruct-v0.3 \
    --adapter mohitxshukla/medqa-mistral-7b-qlora --n 200 --examples 5
# paste eval/results_template.md numbers into model_card/README.md
huggingface-cli upload mohitxshukla/medqa-mistral-7b-qlora model_card/README.md README.md
```

### 4. Serve locally (FastAPI + Docker)
```bash
ADAPTER_REPO=mohitxshukla/medqa-mistral-7b-qlora HF_TOKEN=hf_xxx \
  docker compose -f serving/docker-compose.yml up --build
curl localhost:8000/health
curl -X POST localhost:8000/generate -H 'content-type: application/json' \
  -d '{"question":"What lifestyle changes lower blood pressure?"}'
```

### 5. Deploy the Gradio Space
See `app/README.md` — create a Gradio Space, copy `app.py` + `prompt.py` +
`requirements.txt`, set `ADAPTER_REPO`, request a free ZeroGPU grant.

## Notes / honest constraints
- Free **Colab T4** has no bf16 → training uses fp16. ~8k-example subset keeps it < 4h.
- Free **HF Spaces are CPU-only** → 7B is unusable interactively; use the ZeroGPU
  grant (preferred) or the `USE_GGUF=1` CPU fallback with short outputs.
- **CI cannot train** (no GPU) → it lints + runs a tiny smoke-eval only.
- This is **not medical advice** — disclaimers ship in the card, app, and API.
