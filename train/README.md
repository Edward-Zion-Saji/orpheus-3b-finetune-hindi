# Orpheus TTS — Hindi Male LoRA Finetune

LoRA finetune of [canopylabs/3b-hi-pretrain-research_release](https://huggingface.co/canopylabs/3b-hi-pretrain-research_release) on the [ai4bharat/Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) Hindi Male subset.

The resulting model is published at [edzsaji26/orpheus-3b-0.1-hindi-male-lora](https://huggingface.co/edzsaji26/orpheus-3b-0.1-hindi-male-lora).

## Files

| File | Purpose |
|---|---|
| `prepare_rasa_hindi.py` | Download & tokenise the Rasa Hindi dataset into Orpheus format |
| `config_hindi_lora.yaml` | Training hyperparameters and paths |
| `lora_hindi.py` | LoRA training script |
| `infer_hindi.py` | Inference — generate WAV from Hindi text |

## Setup

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets peft accelerate wandb snac
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu124torch2.6-cp313-cp313-linux_x86_64.whl

huggingface-cli login   # needs access to ai4bharat/Rasa and the base model
wandb login
```

## Step 1 — Prepare dataset

Downloads the Hindi config of `ai4bharat/Rasa`, resamples audio to 24kHz, encodes with SNAC, and saves a tokenised `DatasetDict` to disk.

```bash
python prepare_rasa_hindi.py
# optional:
# python prepare_rasa_hindi.py --max_samples 500 --output_dir /path/to/output
```

**Output format** — each row has three columns:

- `input_ids` — `[128259] + text_tokens + [128009, 128260, 128261, 128257] + audio_tokens + [128258]`
- `labels` — `-100` over the text prefix, audio tokens + stop token for loss
- `attention_mask` — all `1`s (padding handled per-batch in the collator)

Audio token offset: `code + 128266 + position_in_frame * 4096` (7 tokens per SNAC frame).

## Step 2 — Train

```bash
python lora_hindi.py
```

Outputs saved to `save_folder` in `config_hindi_lora.yaml`:
- `adapter_final/` — lightweight PEFT adapter (~2GB)
- `merged/` — full merged model ready for inference (~7GB)

## Step 3 — Inference

```bash
# generate 3 sample WAVs
python infer_hindi.py

# single utterance
python infer_hindi.py --text "आपका स्वागत है" --out output.wav

# custom model
python infer_hindi.py --model_dir /path/to/merged --text "नमस्ते"
```

## Prompt format

```
arjun: <hindi text>
```
