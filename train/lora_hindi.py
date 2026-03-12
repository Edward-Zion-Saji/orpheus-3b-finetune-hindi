import os

import torch
import wandb
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

config_file = "config_hindi_lora.yaml"
with open(config_file) as f:
    config = yaml.safe_load(f)

dataset_path  = config["TTS_dataset"]
model_name    = config["model_name"]
run_name      = config["run_name"]
project_name  = config["project_name"]
save_folder   = config["save_folder"]
epochs        = config["epochs"]
batch_size    = config["batch_size"]
pad_token     = config["pad_token"]
learning_rate = config["learning_rate"]

LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.0


def data_collator(features):
    input_ids      = [torch.tensor(f["input_ids"],      dtype=torch.long) for f in features]
    attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
    labels         = [torch.tensor(f["labels"],         dtype=torch.long) for f in features]
    input_ids      = torch.nn.utils.rnn.pad_sequence(input_ids,      batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels         = torch.nn.utils.rnn.pad_sequence(labels,         batch_first=True, padding_value=-100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    print(f"Loading dataset from {dataset_path} ...")
    ds = load_from_disk(dataset_path)
    train_ds = ds["train"]
    print(f"  train: {len(train_ds)} examples")

    print(f"Loading model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        modules_to_save=["lm_head", "embed_tokens"],
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    wandb.init(project=project_name, name=run_name)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=save_folder,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        report_to="wandb",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    trainer.train()

    adapter_dir = os.path.join(save_folder, "adapter_final")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir}")

    merged = model.merge_and_unload()
    merged_dir = os.path.join(save_folder, "merged")
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
