import argparse
import time
import wave

import numpy as np
import torch
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR    = "./checkpoints_hindi_lora/merged"
VOICE_NAME   = "arjun"
AUDIO_OFFSET = 128266
STOP_TOKEN   = 128258
SAMPLE_RATE  = 24_000
START_TOKEN      = 128259
EOT_TOKEN        = 128009
AUDIO_HEADER_TOK = 128260
AUDIO_DATA_TOK   = 128261
BEGIN_AUDIO_TOK  = 128257

SAMPLE_TEXTS = [
    "नमस्ते, मेरा नाम अर्जुन है और मैं दिल्ली में रहता हूँ। मुझे हिंदी में बात करना बहुत पसंद है क्योंकि यह मेरी मातृभाषा है।",
    "आज सुबह से ही मौसम बहुत सुहाना है और आसमान में हल्के बादल छाए हुए हैं। ऐसे मौसम में चाय पीते हुए किताब पढ़ना बहुत अच्छा लगता है।",
    "भारत एक विविधताओं से भरा हुआ देश है जहाँ अलग-अलग भाषाएँ, संस्कृतियाँ और परंपराएँ एक साथ फलती-फूलती हैं। यहाँ के लोग मिलजुल कर रहते हैं और एक-दूसरे की मदद करते हैं।",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",   type=str,   default=MODEL_DIR)
    p.add_argument("--text",        type=str,   default=None)
    p.add_argument("--out",         type=str,   default=None)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_p",       type=float, default=0.9)
    p.add_argument("--max_tokens",  type=int,   default=2000)
    p.add_argument("--rep_penalty", type=float, default=1.1)
    return p.parse_args()


def build_prompt(text, tokenizer):
    text_ids = tokenizer(f"{VOICE_NAME}: {text}", return_tensors="pt").input_ids
    return torch.cat([
        torch.tensor([[START_TOKEN]]),
        text_ids,
        torch.tensor([[EOT_TOKEN, AUDIO_HEADER_TOK, AUDIO_DATA_TOK, BEGIN_AUDIO_TOK]]),
    ], dim=1)


def tokens_to_audio(token_ids, snac_model, device):
    n = len(token_ids) // 7
    token_ids = token_ids[:n * 7]
    c0, c1, c2 = [], [], []
    for f in range(n):
        i = f * 7
        c0.append(token_ids[i]   - AUDIO_OFFSET)
        c1.append(token_ids[i+1] - AUDIO_OFFSET - 4096)
        c2.append(token_ids[i+2] - AUDIO_OFFSET - 2*4096)
        c2.append(token_ids[i+3] - AUDIO_OFFSET - 3*4096)
        c1.append(token_ids[i+4] - AUDIO_OFFSET - 4*4096)
        c2.append(token_ids[i+5] - AUDIO_OFFSET - 5*4096)
        c2.append(token_ids[i+6] - AUDIO_OFFSET - 6*4096)
    codes = [torch.tensor(c).unsqueeze(0).to(device) for c in [c0, c1, c2]]
    with torch.inference_mode():
        audio = snac_model.decode(codes)
    return audio.squeeze().cpu().numpy()


def save_wav(waveform, path):
    int16 = (waveform * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(int16.tobytes())


def synthesise(text, model, tokenizer, snac_model, device, args):
    input_ids = build_prompt(text, tokenizer).to(device)
    t0 = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            eos_token_id=STOP_TOKEN,
        )
    t1 = time.monotonic()
    new_tokens = generated[0, input_ids.shape[1]:].tolist()
    audio_tokens = [t for t in new_tokens if t != STOP_TOKEN]
    waveform = tokens_to_audio(audio_tokens, snac_model, device)
    duration = len(waveform) / SAMPLE_RATE
    print(f"  {duration:.2f}s in {t1-t0:.2f}s (RTF {(t1-t0)/duration:.2f}x)")
    return waveform


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)

    texts = [(args.text, args.out or "output.wav")] if args.text else [
        (t, f"hindi_sample_{i}.wav") for i, t in enumerate(SAMPLE_TEXTS)
    ]

    for text, out_path in texts:
        print(f"\n{text}")
        save_wav(synthesise(text, model, tokenizer, snac_model, device, args), out_path)
        print(f"  → {out_path}")


if __name__ == "__main__":
    main()
