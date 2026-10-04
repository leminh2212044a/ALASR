"""
Do THOI GIAN THUC TE -> tinh lai ngan sach GPU.

CACH LAM THONG MINH: KHONG train day du o moi muc du lieu.
Thoi gian MOI BUOC gan nhu khong doi theo kich thuoc dataset
(no chi phu thuoc batch size va do dai audio).
Thu thay doi la SO BUOC can thiet.

=> Chi can do sec/step bang mot lan chay ngan (~200 buoc), roi ngoai suy.
   Tiet kiem tu ~8 gio xuong ~10 phut.

CACH DUNG
---------
    python scripts/measure_time.py \
        --train-manifest    data/librispeech/train-clean-100.manifest.jsonl \
        --train-transcripts data/librispeech/train-clean-100.transcripts.jsonl \
        --probe-steps 200
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.w2v2_ctc import finetune, load_manifest, load_transcripts
from scripts.anchor_run import sample_by_hours

# Quy tac tinh so buoc trong w2v2_ctc.finetune():  clip(2000 * hours, 2000, 20000)
def steps_for(hours):
    return int(min(max(2000 * hours, 2000), 20000))


def main(a):
    mf = load_manifest(a.train_manifest)
    tx = load_transcripts(a.train_transcripts)

    # --- Do sec/step bang mot lan chay ngan tren 2 gio du lieu ---
    ids, used = sample_by_hours(mf, 2.0, seed=0)
    print(f"\n### Chay do dac: {a.probe_steps} buoc tren {used:.2f} gio du lieu ###")
    t0 = time.time()
    finetune(ids, mf, tx, out_dir="runs/timing_probe",
             max_steps=a.probe_steps, seed=0)
    probe_sec = time.time() - t0

    # Tru overhead khoi tao (nap model, build dataset) ~ 90 giay
    OVERHEAD = 90
    sec_per_step = max((probe_sec - OVERHEAD) / a.probe_steps, 0.01)

    # --- Do toc do decode tren mot mau nho ---
    print("\n### Do toc do decode ###")
    from models.w2v2_ctc import evaluate
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained("runs/timing_probe")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base",
                                           vocab_size=len(proc.tokenizer),
                                           pad_token_id=proc.tokenizer.pad_token_id)
    sample_ids = list(mf.keys())[:100]
    t1 = time.time()
    evaluate(model, proc, sample_ids, mf, tx, batch_log=10_000)
    sec_per_utt_decode = (time.time() - t1) / len(sample_ids)

    # --- Ngoai suy ---
    rows = []
    for h in [1, 2, 3, 4, 5, 6, 7, 8]:
        st = steps_for(h)
        rows.append({"hours": h, "steps": st,
                     "train_minutes": round(st * sec_per_step / 60, 1)})

    one_run_train_h = sum(r["steps"] for r in rows) * sec_per_step / 3600
    # Moi vong con decode test set (~2620 utterance cua test-clean)
    one_run_eval_h = 8 * 2620 * sec_per_utt_decode / 3600
    one_run_h = one_run_train_h + one_run_eval_h
    total = one_run_h * 15          # 5 strategy x 3 seed

    print()
    print("=" * 64)
    print("  KET QUA DO THOI GIAN")
    print("=" * 64)
    print(f"  Thoi gian moi buoc train : {sec_per_step:.2f} s")
    print(f"  Thoi gian decode moi cau : {sec_per_utt_decode:.3f} s")
    print("-" * 64)
    print("  Ngoai suy thoi gian train tung vong:")
    for r in rows:
        print(f"     vong {r['hours']} gio | {r['steps']:>6} buoc | "
              f"{r['train_minutes']:>7.1f} phut")
    print("-" * 64)
    print(f"  Mot lan chay AL 8 vong  : {one_run_h:.1f} GPU-hour")
    print(f"     trong do train       : {one_run_train_h:.1f} h")
    print(f"     trong do decode      : {one_run_eval_h:.1f} h")
    print(f"  5 strategy x 3 seed     : {total:.0f} GPU-hour")
    print("-" * 64)
    print(f"  Ke hoach ban dau        : 195 GPU-hour")
    if total <= 220:
        print("  KET LUAN: KHA THI, giu nguyen pham vi.")
    elif total <= 350:
        print("  KET LUAN: HOI CAO. Dieu chinh:")
        print("     - Giam so vong 8 -> 6")
        print("     - Giam seed 3 -> 2 cho CoreSet va Longest/Shortest")
        print("     - Danh gia tren tap con test-clean (vd 1000 cau) thay vi toan bo")
    else:
        print("  KET LUAN: VUOT NGAN SACH. Cat pham vi NGAY tuan nay:")
        print("     - Giam pool xuong 50 gio")
        print("     - Giam so vong 8 -> 5")
        print("     - Bo BADGE khoi danh sach baseline")
        print("     - Giam max_steps trong w2v2_ctc.finetune()")
    print("=" * 64)

    Path("runs").mkdir(exist_ok=True)
    with open("runs/timing.json", "w") as f:
        json.dump({
            "sec_per_step": round(sec_per_step, 3),
            "sec_per_utt_decode": round(sec_per_utt_decode, 4),
            "per_round": rows,
            "one_run_gpu_hours": round(one_run_h, 2),
            "estimated_total_gpu_hours": round(total, 1),
        }, f, indent=2)
    print("  -> runs/timing.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--train-transcripts", required=True)
    p.add_argument("--probe-steps", type=int, default=200)
    main(p.parse_args())
