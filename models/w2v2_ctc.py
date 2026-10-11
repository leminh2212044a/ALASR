"""
Fine-tune wav2vec2-base + CTC ky tu.  (BAN SUA — khop voi wav2vec 2.0 muc 4.3)

Module nay duoc goi lai MOI VONG trong vong lap AL (tuan 4), nen giao dien
duoc thiet ke nhu mot HAM:

    model, processor = finetune(labeled_ids, manifest, transcripts, out_dir)
    res = evaluate(model, processor, test_ids, test_manifest, test_transcripts)

=====================================================================
NHUNG GI DA SUA SO VOI BAN TRUOC (va ly do, theo bai goc muc 4.3)
=====================================================================
1. learning_rate: 3e-4  ->  3e-5
   Bai goc dung 2e-5 / 3e-5 cho cac tap con Libri-light (10 phut .. 100 gio).
   Con so 1e-4 trong bai la cho tap 960 gio CO NHAN, khong phai 1 gio.
   LR cao gap 10 lan la nguyen nhan loss TANG ngay sau khi LR dat dinh.

2. Lich hoc: them giai doan GIU NGUYEN (tri-state).
   Bai goc: warmup 10% -> giu hang so 40% -> giam tuyen tinh 50%.
   Ban truoc thieu giai doan giu, LR giam ve ~0 truoc khi model kip hoc.

3. Them giai doan CHI HUAN LUYEN OUTPUT CLASSIFIER.
   Bai goc: "For the first 10k updates only the output classifier is trained."
   lm_head khoi tao ngau nhien -> neu mo Transformer tu buoc 0 thi gradient
   nhieu se pha vo bieu dien da pre-train. Day la nguyen nhan loss dao dong
   quanh 3.0 (ln(30) = 3.40 = doan bua) thay vi giam deu.
   Ty le 15% tong so buoc (3000 buoc -> mo o buoc 450), ty le hoa tu
   10k/320k cua bai goc.

4. ctc_zero_infinity=True — mot vai cau co label dai hon so frame đầu ra se
   cho loss = inf, keo ca batch thanh NaN. Bo qua rieng cac cau do.

5. group_by_length=True — gom cau dai gan nhau, giam padding vo ich.

6. gradient_checkpointing=False — ban truoc bat qua config, cham ~30%.

7. Wav2Vec2FeatureExtractor.from_pretrained(BASE_MODEL) thay vi tu dung tay,
   de do_normalize / return_attention_mask khop dung checkpoint.

8. Bo processor.as_target_processor() (da deprecated) -> dung
   tokenizer(...) va tokenizer.pad(...) truc tiep.

=====================================================================
DIEM LECH CON LAI — PHAI GHI VAO BAO CAO LA RANG BUOC PHAN CUNG
=====================================================================
Bai goc: batch hieu dung 1.600 giay audio (8 GPU).
O day:   per_device 4 x grad_accum 4 x ~12.5s  =  ~200 giay  (1 GPU).
Nho hon ~8 lan. Khong the sua bang code; chi co the ghi nhan.
Hau qua: WER neo co the cao hon 24.5% cua Table 9 vai diem — day la lech
do phan cung, khong phai loi pipeline.

=====================================================================
BA DIEM KY THUAT BAT BUOC (giu nguyen tu ban truoc)
=====================================================================
1. Dung checkpoint 'facebook/wav2vec2-base' (KHONG phai 'wav2vec2-base-960h'
   -- ban do da fine-tune san tren 960 gio, dung no thi ket qua vo nghia).
2. Goi freeze_feature_encoder() -- CNN encoder KHONG duoc cap nhat.
   Day cung la co so cho lap luan "embedding on dinh" o tuan 6-7.
3. Vocab KY TU (~30 token). Khong dung subword: gradient embedding cua BADGE
   se phinh tu 23k len 768k chieu moi frame.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import Dataset
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import jiwer

BASE_MODEL = "facebook/wav2vec2-base"
SAMPLE_RATE = 16_000

# --- Sieu tham so fine-tune (theo wav2vec 2.0 muc 4.3) ---
LEARNING_RATE = 3e-5        # bai goc: 2e-5 hoac 3e-5 cho cac tap con Libri-light
WARMUP_RATIO = 0.10         # tri-state: warmup 10%
HOLD_RATIO = 0.40           #            giu hang so 40%
FREEZE_RATIO = 0.15         # % buoc dau CHI huan luyen lm_head
PER_DEVICE_BATCH = 4        # T4: 4  |  L4 / V100: 8  |  A100: 16
GRAD_ACCUM = 4              # batch hieu dung = PER_DEVICE_BATCH * GRAD_ACCUM
DEFAULT_MAX_STEPS = 3000


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
    vocab["[PAD]"] = len(vocab)      # CTC blank

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  vocab: {len(vocab)} token -> {out_path}")
    return out_path


def make_processor(vocab_path: Path, base_model: str = BASE_MODEL) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
    )
    # Lay feature extractor TU CHECKPOINT -> do_normalize va
    # return_attention_mask tu dong dung voi wav2vec2-base.
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# =====================================================================
# 2. DATASET
# =====================================================================

def normalize_text(t: str) -> str:
    """Chuan hoa transcript: viet hoa, thay dau cach bang '|'."""
    return t.upper().replace(" ", "|")


def build_dataset(utt_ids, manifest: Dict, transcripts: Dict, processor):
    rows = [{"path": manifest[uid]["path"], "text": transcripts[uid]}
            for uid in utt_ids]
    ds = Dataset.from_list(rows)

    def _prepare(batch):
        audio, sr = sf.read(batch["path"])
        assert sr == SAMPLE_RATE, f"Sampling rate {sr} != {SAMPLE_RATE}"
        batch["input_values"] = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE
        ).input_values[0]
        # Cot nay cho group_by_length dung de gom cau dai gan nhau
        batch["input_length"] = len(batch["input_values"])
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

        batch = self.processor.feature_extractor.pad(
            inputs, padding=True, return_tensors="pt")
        lab_batch = self.processor.tokenizer.pad(
            labels, padding=True, return_tensors="pt")

        # -100 de CTC loss bo qua vi tri padding
        batch["labels"] = lab_batch["input_ids"].masked_fill(
            lab_batch.attention_mask.ne(1), -100
        )
        return batch


# =====================================================================
# 3. LICH HOC TRI-STATE  +  CALLBACK MO DONG TRANSFORMER
# =====================================================================

def tri_state_schedule(optimizer, total_steps,
                       warmup=WARMUP_RATIO, hold=HOLD_RATIO):
    """
    wav2vec 2.0 muc 4.3: "a tri-state rate schedule where the learning rate is
    warmed up for the first 10% of updates, held constant for the next 40% and
    then linearly decayed for the remainder."
    """
    w = max(int(total_steps * warmup), 1)
    h = max(int(total_steps * (warmup + hold)), w + 1)

    def fn(step):
        if step < w:
            return step / w                                 # tang dan
        if step < h:
            return 1.0                                      # giu nguyen
        return max(0.0, (total_steps - step) / max(total_steps - h, 1))

    return LambdaLR(optimizer, fn)


class FreezeTransformerCallback(TrainerCallback):
    """
    wav2vec 2.0 muc 4.3: "For the first 10k updates only the output classifier
    is trained, after which the Transformer is also updated."

    lm_head moi khoi tao ngau nhien. Neu mo Transformer ngay tu buoc 0,
    gradient nhieu se pha vo bieu dien da pre-train -> loss dao dong.
    """

    def __init__(self, model, unfreeze_at: int):
        self.model = model
        self.unfreeze_at = unfreeze_at
        for p in model.wav2vec2.parameters():
            p.requires_grad = False
        print(f"    dong bang Transformer, se mo o buoc {unfreeze_at}")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == self.unfreeze_at:
            for p in self.model.wav2vec2.parameters():
                p.requires_grad = True
            self.model.freeze_feature_encoder()   # CNN encoder VAN dong bang
            print(f"\n    [buoc {state.global_step}] mo dong Transformer "
                  f"(feature encoder giu nguyen dong bang)\n")


# =====================================================================
# 4. FINE-TUNE
# =====================================================================

def finetune(utt_ids, manifest, transcripts, out_dir,
             max_steps=None, seed=0, base_model=BASE_MODEL,
             learning_rate=LEARNING_RATE,
             per_device_batch=PER_DEVICE_BATCH,
             grad_accum=GRAD_ACCUM):
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

    if max_steps is None:
        max_steps = int(np.clip(2000 * hours, DEFAULT_MAX_STEPS, 20_000))
    unfreeze_at = max(int(max_steps * FREEZE_RATIO), 1)

    print(f"    max_steps      = {max_steps}")
    print(f"    learning_rate  = {learning_rate:g}  (bai goc: 2e-5 / 3e-5)")
    print(f"    batch hieu dung= {per_device_batch} x {grad_accum} "
          f"= {per_device_batch * grad_accum} cau "
          f"(bai goc: 1600 giay tren 8 GPU)")
    print(f"    mo Transformer o buoc {unfreeze_at} "
          f"({int(FREEZE_RATIO * 100)}% so buoc)")

    vocab_path = out_dir / "vocab.json"
    build_vocab([transcripts[u] for u in utt_ids], vocab_path)
    processor = make_processor(vocab_path, base_model)
    processor.save_pretrained(out_dir)

    ds = build_dataset(utt_ids, manifest, transcripts, processor)

    model = Wav2Vec2ForCTC.from_pretrained(
        base_model,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,             # bo qua cau co loss = inf
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        # SpecAugment -- bai goc ap dung tren bieu dien cua feature encoder
        mask_time_prob=0.065,
        mask_time_length=10,
        mask_feature_prob=0.0,
        layerdrop=0.05,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        activation_dropout=0.1,
    )
    model.freeze_feature_encoder()          # <-- BAT BUOC

    # --- Optimizer + lich hoc tu dinh nghia (khong de Trainer tu tao) ---
    optimizer = AdamW(model.parameters(), lr=learning_rate,
                      betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0)
    scheduler = tri_state_schedule(optimizer, max_steps)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        # KHONG dat learning_rate / warmup_ratio / lr_scheduler_type o day:
        # da truyen optimizer + scheduler truc tiep vao Trainer.
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=False,       # nhanh hon ~30%
        group_by_length=True,               # giam padding vo ich
        length_column_name="input_length",
        save_strategy="no",
        logging_steps=50,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=seed,
        report_to=[],                       # doi thanh ["wandb"] tu tuan 6
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds,
        data_collator=DataCollatorCTC(processor=processor),
        tokenizer=processor.feature_extractor,
        optimizers=(optimizer, scheduler),
        callbacks=[FreezeTransformerCallback(model, unfreeze_at)],
    )
    trainer.train()
    return model, processor


# =====================================================================
# 5. DANH GIA
# =====================================================================

@torch.no_grad()
def evaluate(model, processor, utt_ids, manifest, transcripts,
             batch_log=200, n_examples=5):
    """Greedy decode, KHONG dung LM. Tra ve WER va CER."""
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    refs, hyps = [], []
    for i, uid in enumerate(utt_ids, 1):
        audio, sr = sf.read(manifest[uid]["path"])
        inputs = processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        logits = model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(pred_ids)[0]

        hyps.append(hyp.strip())
        refs.append(transcripts[uid].upper().strip())
        if i % batch_log == 0:
            print(f"    decode {i}/{len(utt_ids)} ...")

    # --- In vai vi du de kiem tra bang mat ---
    if n_examples:
        print("\n    --- vi du decode ---")
        for r, h in list(zip(refs, hyps))[:n_examples]:
            print(f"    REF: {r[:90]}")
            print(f"    HYP: {h[:90]}")
            print()

    # --- Canh bao blank collapse: output rong = model chi doan blank ---
    n_empty = sum(1 for h in hyps if not h)
    if n_empty:
        pct = 100 * n_empty / len(hyps)
        print(f"    CANH BAO: {n_empty}/{len(hyps)} ({pct:.1f}%) cau cho "
              f"output RONG -> blank collapse. "
              f"CER ~100% nghia la model chua hoc duoc gi.")

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    return {"wer": wer * 100, "cer": cer * 100,
            "n_utts": len(utt_ids), "n_empty_hyp": n_empty}


# =====================================================================
# 6. TIEN ICH DOC FILE
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