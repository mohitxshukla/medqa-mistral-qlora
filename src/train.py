"""QLoRA fine-tuning of Mistral-7B-Instruct on medical Q&A.

CLI mirror of ``notebooks/train_qlora_colab.ipynb``. Runs on a single 16GB GPU
(free Colab T4). Loads the base model in 4-bit nf4, attaches a LoRA adapter, and
trains with TRL's SFTTrainer.

Example:
    python src/train.py \
        --base mistralai/Mistral-7B-Instruct-v0.3 \
        --output-dir out/medqa-mistral-7b-qlora \
        --push-to-hub mohitxshukla/medqa-mistral-7b-qlora
"""

from __future__ import annotations

import argparse

import torch
from datasets import DatasetDict
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from data import SplitConfig, load_splits

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
# LoRA on all attention + MLP projections — standard QLoRA coverage.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune Mistral-7B on medical Q&A.")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--output-dir", default="out/medqa-mistral-7b-qlora")
    ap.add_argument("--train-size", type=int, default=8000)
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--push-to-hub", default=None, help="HF repo id, e.g. user/medqa-mistral-7b-qlora")
    return ap.parse_args()


def load_model_and_tokenizer(base: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4 has no bf16
    )
    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model.config.pretraining_tp = 1

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return model, tok


def main() -> None:
    args = build_args()

    ds: DatasetDict = load_splits(SplitConfig(train_size=args.train_size, eval_size=args.eval_size))
    model, tok = load_model_and_tokenizer(args.base)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_seq_len,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds["train"],
        peft_config=peft_config,
        processing_class=tok,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.push_to_hub:
        trainer.model.push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)
        print(f"Pushed adapter to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
