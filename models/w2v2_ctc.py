"""
Fine-tune wav2vec2-base + CTC ky tu.

Module nay se duoc goi lai MOI VONG trong vong lap AL (tuan 4),
nen giao dien duoc thiet ke nhu mot HAM:

    model, processor = finetune(labeled, manifest, out_dir, hours=1.0)
    wer = evaluate(model, processor, test_manifest, test_transcripts)

BA DIEM KY THUAT BAT BUOC
-------------------------
1. Dung checkpoint 'facebook/wav2vec2-base'  (KHONG phai 'wav2vec2-base-960h'
   -- ban do da fine-tune san tren 960 gio, dung no thi ket qua vo nghia).
2. Goi freeze_feature_encoder() -- CNN encoder KHONG duoc cap nhat.
   Day cung la co so cho lap luan "embedding on dinh" o tuan 6-7.
3. Vocab KY TU (~32 token). Khong dung subword: gradient embedding cua BADGE
   se phinh tu 23k len 768k chieu moi frame.
"""

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import soundfile as sf
import torch
from datasets import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
)
import jiwer

BASE_MODEL = "facebook/wav2vec2-base"
SAMPLE_RATE = 16_000


# =====================================================================
# 1. VOCAB & PROCESSOR
# =====================================================================

def build_vocab(texts: List[str], out_path: Path) -> Path:
    """Tao vocab ky tu tu tap transcript. LibriSpeech chi co A-Z, dau nhay, space."""
    chars = set()
    for t in texts:
        chars.update(t.upper())
    chars.discard(" ")

    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["|"] = len(vocab)          # word delimiter (thay cho dau cach)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  vocab: {len(vocab)} token -> {out_path}")
    return out_path


def make_processor(vocab_path: Path) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0,
        do_normalize=True, return_attention_mask=False,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# =====================================================================
# 2. DATASET
# =====================================================================

def normalize_text(t: str) -> str:
    """Chuan hoa transcript: viet hoa, thay dau cach bang '|'."""
    return t.upper().replace(" ", "|")


def build_dataset(utt_ids, manifest: Dict, transcripts: Dict, processor):
    rows = []
    for uid in utt_ids:
        rows.append({"path": manifest[uid]["path"], "text": transcripts[uid]})
    ds = Dataset.from_list(rows)

    def _prepare(batch):
        audio, sr = sf.read(batch["path"])
        assert sr == SAMPLE_RATE, f"Sampling rate {sr} != {SAMPLE_RATE}"
        batch["input_values"] = processor(
            audio, sampling_rate=SAMPLE_RATE
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])          # cho group_by_length
        batch["labels"] = processor.tokenizer(
            normalize_text(batch["text"])
        ).input_ids
        return batch

    return ds.map(_prepare, remove_columns=ds.column_names, num_proc=1)


@dataclass
class DataCollatorCTC:
    processor: Wav2Vec2Processor

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        inputs = [{"input_values": f["input_values"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(inputs, padding=True, return_tensors="pt")
        lab_batch = self.processor.tokenizer.pad(
            labels, padding=True, return_tensors="pt"
        )

        # -100 de CTC loss bo qua vi tri padding
        batch["labels"] = lab_batch["input_ids"].masked_fill(
            lab_batch.attention_mask.ne(1), -100
        )
        return batch


# =====================================================================
# 3. FINE-TUNE
# =====================================================================

def finetune(utt_ids, manifest, transcripts, out_dir,
             max_steps=None, seed=0, base_model=BASE_MODEL):
    """
    utt_ids     : danh sach utt_id DA CO NHAN
    manifest    : dict[utt_id] -> {path, duration, ...}
    transcripts : dict[utt_id] -> text
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    hours = sum(manifest[u]["duration"] for u in utt_ids) / 3600
    print(f"\n=== FINE-TUNE: {len(utt_ids)} utterance = {hours:.2f} gio ===")

    # So buoc huan luyen ti le voi luong du lieu.
    # Tham khao wav2vec 2.0 Table 6; con so duoi la diem xuat phat, can tinh chinh.
    if max_steps is None:
        eff_batch = 16                      # per_device_batch x grad_accum
        max_steps = int(np.clip(55 * len(utt_ids) / eff_batch, 800, 8000))
    print(f"    max_steps = {max_steps}  (~{max_steps*16/len(utt_ids):.0f} epoch)")

    vocab_path = out_dir / "vocab.json"
    build_vocab([transcripts[u] for u in utt_ids], vocab_path)
    processor = make_processor(vocab_path)
    processor.save_pretrained(out_dir)

    ds = build_dataset(utt_ids, manifest, transcripts, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        base_model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True, 
        gradient_checkpointing=False,  
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        # SpecAugment -- wav2vec 2.0 ghi ro no cai thien dang ke o vung it nhan
        mask_time_prob=0.065,
        mask_time_length=10,
        mask_feature_prob=0.0,
        layerdrop=0.05,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        activation_dropout=0.1,
    )
    model.freeze_feature_encoder()          # <-- BAT BUOC

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        learning_rate=3e-4,
        warmup_ratio=0.1,                   # lich tri-state cua wav2vec 2.0
        lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=100,
        group_by_length=True,
        length_column_name="input_length",
        seed=seed,
        report_to=[],                       # doi thanh ["wandb"] tu tuan 6
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds,
        data_collator=DataCollatorCTC(processor=processor),
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    return model, processor


# =====================================================================
# 4. DANH GIA
# =====================================================================

@torch.no_grad()
def evaluate(model, processor, utt_ids, manifest, transcripts, batch_log=200):
    """Greedy decode, KHONG dung LM. Tra ve WER va CER."""
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    refs, hyps = [], []
    for i, uid in enumerate(utt_ids, 1):
        audio, sr = sf.read(manifest[uid]["path"])
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        logits = model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0]

        hyps.append(hyp.strip())
        refs.append(transcripts[uid].upper().strip())
        if i % batch_log == 0:
            print(f"    decode {i}/{len(utt_ids)} ...")

    print("\n--- 5 vi du dau tien ---")
    for r, h in zip(refs[:5], hyps[:5]):
        print(f"  REF: {r[:70]}")
        print(f"  HYP: '{h[:70]}'")
        print()

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return {"wer": wer * 100, "cer": cer * 100, "n_utts": len(utt_ids)}


# =====================================================================
# 5. TIEN ICH DOC FILE
# =====================================================================

def load_manifest(path) -> Dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["utt_id"]] = r
    return out


def load_transcripts(path) -> Dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["utt_id"]] = r["text"]
    return out