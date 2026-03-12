import argparse
import io
import os

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from datasets import Audio, Dataset, DatasetDict, load_dataset
from snac import SNAC
from transformers import AutoTokenizer

TOKENIZER     = "canopylabs/3b-hi-pretrain-research_release"
VOICE_NAME    = "arjun"
AUDIO_OFFSET  = 128266
SNAC_SR       = 24_000
MAX_SEQ_LEN   = 8192

START_TOKEN      = 128259
EOT_TOKEN        = 128009
AUDIO_HEADER_TOK = 128260
AUDIO_DATA_TOK   = 128261
BEGIN_AUDIO_TOK  = 128257
STOP_TOKEN       = 128258


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",    type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--output_dir",  type=str, default="./rasa_hindi_orpheus")
    return p.parse_args()


def get_hf_token(cli_token):
    token = cli_token or os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass
    if not token:
        raise ValueError("No HuggingFace token found. Run `huggingface-cli login`.")
    return token


def load_snac(device):
    return SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)


def encode_audio(snac_model, audio_np, device):
    tensor = torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        codes = snac_model.encode(tensor)
    c0 = codes[0].squeeze().tolist()
    c1 = codes[1].squeeze().tolist()
    c2 = codes[2].squeeze().tolist()
    if isinstance(c0, int): c0 = [c0]
    if isinstance(c1, int): c1 = [c1]
    if isinstance(c2, int): c2 = [c2]
    return c0, c1, c2


def codes_to_tokens(c0, c1, c2):
    n = len(c0)
    tokens = []
    for f in range(n):
        for pos, code in enumerate([
            c0[f],
            c1[f * 2],
            c2[f * 4],
            c2[f * 4 + 1],
            c1[f * 2 + 1],
            c2[f * 4 + 2],
            c2[f * 4 + 3],
        ]):
            tokens.append(code + AUDIO_OFFSET + pos * 4096)
    return tokens


def build_sequence(text, audio_token_ids, tokenizer):
    prompt = f"{VOICE_NAME}: {text}"
    text_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    prefix = [START_TOKEN] + text_tokens + [EOT_TOKEN, AUDIO_HEADER_TOK, AUDIO_DATA_TOK, BEGIN_AUDIO_TOK]
    suffix = audio_token_ids + [STOP_TOKEN]
    if len(prefix) + len(suffix) > MAX_SEQ_LEN:
        return None
    return {
        "input_ids":      prefix + suffix,
        "attention_mask": [1] * (len(prefix) + len(suffix)),
        "labels":         [-100] * len(prefix) + suffix,
    }


def process_split(ds, snac_model, tokenizer, device, max_samples=None):
    records, skipped = [], 0
    for i, example in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        if i % 200 == 0:
            print(f"  {i} ...")
        try:
            audio_info = example["audio"]
            audio_bytes = audio_info.get("bytes") or open(audio_info["path"], "rb").read()
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != SNAC_SR:
                waveform = AF.resample(waveform, sr, SNAC_SR)
            audio_np = waveform.squeeze(0).numpy()
            c0, c1, c2 = encode_audio(snac_model, audio_np, device)
            record = build_sequence(example["text"].strip(), codes_to_tokens(c0, c1, c2), tokenizer)
            if record is None:
                skipped += 1
            else:
                records.append(record)
        except Exception:
            skipped += 1
    print(f"  kept {len(records)}, skipped {skipped}")
    return records


def main():
    args = parse_args()
    hf_token = get_hf_token(args.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset ...")
    raw = load_dataset("ai4bharat/Rasa", name="Hindi", token=hf_token)
    for split in raw:
        raw[split] = raw[split].cast_column("audio", Audio(decode=False))
        raw[split] = raw[split].filter(
            lambda ex: ex["gender"].strip().lower() == "male",
            desc=f"Filtering {split}",
        )
        print(f"  {split}: {len(raw[split])} examples")

    print("Loading SNAC ...")
    snac_model = load_snac(device)

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, token=hf_token)

    splits = {}
    for split, ds in raw.items():
        print(f"\nProcessing {split} ...")
        max_s = args.max_samples if split == "train" else None
        records = process_split(ds, snac_model, tokenizer, device, max_s)
        if records:
            splits[split] = Dataset.from_list(records)

    os.makedirs(args.output_dir, exist_ok=True)
    DatasetDict(splits).save_to_disk(args.output_dir)
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
