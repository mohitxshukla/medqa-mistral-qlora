---
base_model: mistralai/Mistral-7B-Instruct-v0.3
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - medical
  - question-answering
  - qlora
  - lora
  - peft
  - mistral
datasets:
  - lavita/ChatDoctor-HealthCareMagic-100k
language:
  - en
---

# MedQA-Mistral-7B-QLoRA

A QLoRA **LoRA adapter** for `mistralai/Mistral-7B-Instruct-v0.3`, fine-tuned on
public medical Q&A to give clearer, more direct patient-facing answers.

> ⚠️ **Educational information only — not medical advice.** This model does not
> diagnose, prescribe, or replace a licensed clinician. Do not use for clinical
> decision-making.

## Model details

- **Base model:** `mistralai/Mistral-7B-Instruct-v0.3` (loaded 4-bit nf4 for training)
- **Adapter type:** LoRA (PEFT) — only adapter weights are published here (~tens of MB)
- **Task:** English medical question answering
- **License:** Apache-2.0 (inherits base + dataset terms)

## Intended use & limitations

**Intended:** demos, education, prototyping medical-information assistants, and as
a portfolio example of domain QLoRA fine-tuning.

**Not intended / limitations:**
- Not a medical device; can hallucinate or give outdated/incorrect information.
- Trained on informal online doctor-answer text — style and depth vary.
- English only; no patient context, no follow-up reasoning over records.
- No safety RLHF beyond the base model — verify before relying on any output.

## Training data

[`lavita/ChatDoctor-HealthCareMagic-100k`](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
— patient questions paired with doctor responses. We subsample ~8,000 deduped
examples for training and hold out 500 for evaluation. Each row is formatted as a
Mistral instruction:

```
<s>[INST] {system prompt}

{patient question} [/INST] {doctor answer}</s>
```

## Training configuration

| Setting | Value |
|---|---|
| Method | QLoRA (4-bit nf4 + double quant) |
| LoRA rank / α / dropout | 16 / 32 / 0.05 |
| Target modules | q,k,v,o,gate,up,down proj |
| Epochs | 1 |
| Batch size × grad-accum | 2 × 8 (effective 16) |
| Learning rate / scheduler | 2e-4 / cosine, 3% warmup |
| Optimizer | paged_adamw_8bit |
| Precision | fp16, gradient checkpointing |
| Max seq length | 1024 |
| Hardware | 1× T4 16GB (free Colab) |

## Evaluation (before vs after)

Held-out 200 examples. Metrics produced by `src/evaluate.py`.

<!-- Replace TBD with the table generated into eval/results_template.md after your run. -->

| Metric | Base | Fine-tuned | Better |
|---|---|---|---|
| Perplexity ↓ | TBD | TBD | lower |
| ROUGE-L ↑ | TBD | TBD | higher |
| BERTScore F1 ↑ | TBD | TBD | higher |

_5 side-by-side example generations are appended from the eval run._

## Example prompts

- "What are the early warning signs of type 2 diabetes?"
- "Is it safe to take ibuprofen and paracetamol together?"
- "What lifestyle changes help lower high blood pressure?"

## How to use

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_id = "mohitxshukla/medqa-mistral-7b-qlora"

tok = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, adapter_id)

q = "What are the early warning signs of type 2 diabetes?"
prompt = f"<s>[INST] {q} [/INST]"
ids = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=256, do_sample=True, temperature=0.3)
print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
```

## Reproduce

Code, training notebook, eval, and a Gradio/FastAPI demo:
**https://github.com/mohitxshukla/medqa-mistral-qlora**

## Citation

Built on Mistral-7B-Instruct-v0.3 and the ChatDoctor-HealthCareMagic-100k dataset.
Fine-tuned with QLoRA via Hugging Face PEFT + TRL.
