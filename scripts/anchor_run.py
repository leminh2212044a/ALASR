"""
NEO KET QUA -- cot moc quan trong nhat cua tuan 3.

Fine-tune wav2vec2-base tren DUNG 1 gio LibriSpeech, decode KHONG dung LM,
roi doi chieu voi con so da cong bo trong bai wav2vec 2.0 (Table 9):

        BASE, 1 gio nhan, khong LM  ->  24.5 / 29.7   (test-clean / test-other)

Nguyen tac: LUON TAI LAP DUOC MOT KET QUA DA BIET TRUOC KHI TAO RA KET QUA MOI.

CACH DUNG
---------
    python scripts/anchor_run.py \
        --train-manifest    data/librispeech/train-clean-100.manifest.jsonl \
        --train-transcripts data/librispeech/train-clean-100.transcripts.jsonl \
        --test-manifest     data/librispeech/test-clean.manifest.jsonl \
        --test-transcripts  data/librispeech/test-clean.transcripts.jsonl \
        --hours 1.0 --seed 0 --out-dir runs/anchor
"""

import argparse
import json
import random
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.w2v2_ctc import (
    finetune, evaluate, load_manifest, load_transcripts,
)


def sample_by_hours(manifest, hours, seed=0):
    """Boc ngau nhien cho toi khi du so gio. Day chinh la SEED SET cua vong lap AL."""
    ids = list(manifest.keys())
    random.Random(seed).shuffle(ids)
    picked, used = [], 0.0
    for uid in ids:
        d = manifest[uid]["duration"] / 3600
        if used + d > hours:
            continue                      # cau qua dai -> bo qua, thu cau tiep
        picked.append(uid); used += d
        if hours - used < 0.001:
            break
    return picked, used


def main(a):
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_mf = load_manifest(a.train_manifest)
    train_tx = load_transcripts(a.train_transcripts)
    test_mf = load_manifest(a.test_manifest)
    test_tx = load_transcripts(a.test_transcripts)

    print(f"Pool: {len(train_mf)} utterance")
    print(f"Test: {len(test_mf)} utterance")

    seed_ids, used = sample_by_hours(train_mf, a.hours, a.seed)
    print(f"Seed set: {len(seed_ids)} utterance = {used:.3f} gio (yeu cau {a.hours})")

    t0 = time.time()
    model, processor = finetune(
        seed_ids, train_mf, train_tx,
        out_dir=out_dir, max_steps=a.max_steps, seed=a.seed,
    )
    train_sec = time.time() - t0

    t1 = time.time()
    res = evaluate(model, processor, list(test_mf.keys()), test_mf, test_tx)
    eval_sec = time.time() - t1

    record = {
        "hours_requested": a.hours,
        "hours_actual": round(used, 3),
        "n_train_utts": len(seed_ids),
        "seed": a.seed,
        "max_steps": a.max_steps,
        "wer_nolm": round(res["wer"], 2),
        "cer_nolm": round(res["cer"], 2),
        "train_seconds": round(train_sec, 1),
        "eval_seconds": round(eval_sec, 1),
    }
    with open(out_dir / "anchor_result.json", "w") as f:
        json.dump(record, f, indent=2)

    # ---------------- BAO CAO ----------------
    print()
    print("=" * 62)
    print("  KET QUA NEO")
    print("=" * 62)
    print(f"  Du lieu huan luyen : {used:.2f} gio ({len(seed_ids)} utterance)")
    print(f"  WER (khong LM)     : {res['wer']:.2f} %")
    print(f"  CER (khong LM)     : {res['cer']:.2f} %")
    print(f"  Thoi gian train    : {train_sec/60:.1f} phut")
    print(f"  Thoi gian decode   : {eval_sec/60:.1f} phut")
    print("-" * 62)
    print("  Tham chieu wav2vec 2.0 Table 9 (BASE, 1h, khong LM):")
    print("      test-clean  24.5 %     test-other  29.7 %")
    print("-" * 62)

    w = res["wer"]
    if 20 <= w <= 30:
        print("  KET LUAN: DAT. Pipeline chay dung, chuyen sang tuan 4.")
    elif 30 < w <= 40:
        print("  KET LUAN: CHAP NHAN DUOC nhung hoi cao.")
        print("            Thu tang max_steps hoac tinh chinh learning rate.")
    else:
        print("  KET LUAN: KHONG DAT -- PIPELINE CO LOI. Dung lai, kiem tra:")
        print("     1. Dung 'facebook/wav2vec2-base' chu KHONG phai '...-960h'?")
        print("     2. Sampling rate co dung 16 kHz khong?")
        print("     3. Da goi freeze_feature_encoder() chua?")
        print("     4. Vocab co dung ky tu + '|' lam word delimiter khong?")
        print("     5. Transcript va hypothesis co cung dinh dang hoa/thuong khong?")
        print("     6. max_steps co qua nho khong?")
    print("=" * 62)
    print(f"  -> {out_dir/'anchor_result.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--train-transcripts", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--test-transcripts", required=True)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-dir", default="runs/anchor")
    main(p.parse_args())