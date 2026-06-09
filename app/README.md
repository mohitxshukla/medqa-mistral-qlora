---
title: MedQA Mistral-7B QLoRA
emoji: 🩺
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
models:
  - mistralai/Mistral-7B-Instruct-v0.3
short_description: Medical Q&A demo of a QLoRA-fine-tuned Mistral-7B adapter.
---

# 🩺 MedQA Mistral-7B (QLoRA) — Space

Live demo of a LoRA adapter fine-tuned on public medical Q&A. See the model repo
for training config and before/after evaluation.

## Deploying this Space

The Space root must contain: `app.py`, `prompt.py` (copy from `src/prompt.py`),
and `requirements.txt`.

```bash
# from the repo root, after the adapter is on the Hub:
huggingface-cli repo create medqa-mistral-demo --type space --space_sdk gradio
git clone https://huggingface.co/spaces/mohitxshukla/medqa-mistral-demo space && cd space
cp ../app/app.py ../app/requirements.txt ../src/prompt.py .
cp ../app/README.md .
git add . && git commit -m "Add MedQA Gradio demo" && git push
```

Set the Space **secret/variable** `ADAPTER_REPO=mohitxshukla/medqa-mistral-7b-qlora`.

### GPU
- **Preferred:** request a free **ZeroGPU** grant (Space → Settings → Hardware →
  request grant). `app.py` auto-uses the `@spaces.GPU` decorator.
- **Fallback while pending:** set `USE_GGUF=1`, `GGUF_REPO`, `GGUF_FILE` to run a
  merged q4 GGUF on CPU (slower; keep `MAX_NEW_TOKENS` small).

⚠️ Educational information only — not medical advice.
