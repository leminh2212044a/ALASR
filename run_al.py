"""
run_al.py — VONG LAP ACTIVE LEARNING.

Day la "xuong song" cua ca do an. Moi chien luoc o cac tuan sau chi la mot
module cam vao day; ban than vong lap khong doi nua.

SO DO MOT LAN CHAY
------------------
    vong 0 : seed set 1 gio chon NGAU NHIEN (giong nhau cho moi chien luoc,
             cung seed) -> fine-tune -> do WER         <-- diem goc chung
    vong 1 : strategy xep hang cac cau CHUA nhan
             -> Pool.take_until_budget(+1 gio)
             -> Oracle.query() gan nhan
             -> fine-tune LAI TU DAU tren 2 gio -> do WER
    ...
    vong 7 : tong 8 gio

BA QUYET DINH THIET KE PHAI GHI VAO BAO CAO
-------------------------------------------
1. HUAN LUYEN LAI TU DAU moi vong (tu facebook/wav2vec2-base), KHONG
   huan luyen tiep tu checkpoint vong truoc.
   Ly do: neu huan luyen tiep, WER o vong r phu thuoc ca vao QUY DAO
   toi uu hoa cua cac vong truoc, khong chi vao TAP DU LIEU da chon.
   Khi do khong the ket luan "du lieu chien luoc A chon tot hon B".
   Gia: dat hon, nhung day la chuan trong cac bai AL.

2. SEED SET DUNG CHUNG. Moi chien luoc bat dau tu DUNG 1 gio giong nhau
   (cung seed). Neu seed set khac nhau, chenh lech WER o vong cuoi co the
   chi la do seed set may man.

3. LAY MAU CON TAP UNG VIEN (--candidate-size). Chay suy dien tren ca
   28.000 cau moi vong rat ton GPU. Lay mau con N cau (mac dinh 5000),
   DUNG CHUNG cho moi chien luoc trong cung vong (cung seed) -> van cong
   bang, ma re hon ~6 lan. Random khong bi anh huong; cac chien luoc
   uncertainty bi gioi han pham vi tim kiem nhu nhau.

CACH DUNG
---------
    python run_al.py \
        --train-manifest    data/librispeech/train-clean-100.manifest.jsonl \
        --train-transcripts data/librispeech/train-clean-100.transcripts.jsonl \
        --test-manifest     data/librispeech/test-clean.manifest.jsonl \
        --test-transcripts  data/librispeech/test-clean.transcripts.jsonl \
        --strategy random --rounds 4 --seed-hours 1.0 --budget-hours 1.0 \
        --seed 0 --out-dir runs/al_random_s0
"""

import argparse
import json
import random
import time
from pathlib import Path

from data.oracle import Oracle
from data.pool import Pool
from models.w2v2_ctc import (
    finetune, evaluate, load_manifest, load_transcripts, load_round_model,
)
from strategies import SelectionContext, build


# =====================================================================
# SEED SET — giong nhau cho moi chien luoc
# =====================================================================

def pick_seed_set(pool: Pool, hours: float, seed: int):
    """
    Vong 0 bao gio cung la ngau nhien: chua co model nen khong the tinh
    do khong chac chan. Dung seed GOC (khong + round) de moi chien luoc
    nhan dung mot seed set.
    """
    ids = pool.unlabeled                       # da sort -> tai lap duoc
    random.Random(seed).shuffle(ids)
    return pool.take_until_budget(ids, hours)


# =====================================================================
# VONG LAP CHINH
# =====================================================================

def main(a):
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    train_mf = load_manifest(a.train_manifest)
    test_mf = load_manifest(a.test_manifest)
    test_tx = load_transcripts(a.test_transcripts)

    strategy = build(a.strategy)

    # ---------- CHAY TIEP tu lan chay bi ngat ----------
    # Trang thai pool duoc luu sau MOI vong, nen chi can nap lai file moi nhat.
    start_round, model, processor = 0, None, None
    saved = sorted(out_dir.glob("pool_round*.json"),
                   key=lambda p: int(p.stem.replace("pool_round", "")))

    pool = Pool(train_mf)
    oracle = Oracle(a.train_transcripts,
                    log_path=out_dir / "oracle_log.jsonl",
                    append=bool(a.resume and saved))

    if a.resume and saved:
        last = int(saved[-1].stem.replace("pool_round", ""))
        pool.load(saved[-1])
        oracle.restore(pool.round_map)
        oracle.audit(pool)
        start_round = last + 1
        print(f"\nCHAY TIEP: da xong vong 0..{last} "
              f"({pool.labeled_hours:.2f} gio), bat dau tu vong {start_round}")
        if strategy.needs_model:
            prev = out_dir / f"round{last}"
            print(f"  nap model vong {last} tu {prev}")
            model, processor = load_round_model(prev)
    elif a.resume:
        print("\n--resume nhung chua co trang thai nao -> chay tu vong 0")

    test_ids = sorted(test_mf.keys())
    if a.test_subset:
        random.Random(12345).shuffle(test_ids)   # seed CO DINH -> moi lan chay
        test_ids = sorted(test_ids[:a.test_subset])  # dung dung mot tap test

    print()
    print("=" * 66)
    print(f"  ACTIVE LEARNING  |  chien luoc = {strategy.name}  |  seed = {a.seed}")
    print("=" * 66)
    print(f"  Pool        : {len(train_mf)} cau / {pool.unlabeled_hours:.1f} gio")
    print(f"  Test        : {len(test_ids)} cau")
    print(f"  Seed set    : {a.seed_hours} gio")
    print(f"  Moi vong    : +{a.budget_hours} gio  x  {a.rounds} vong")
    print(f"  Ngan sach cuoi: {a.seed_hours + a.budget_hours * a.rounds} gio")
    print("=" * 66)

    records = []

    for r in range(start_round, a.rounds + 1):
        t_round = time.time()
        print(f"\n{'#' * 66}\n#  VONG {r}\n{'#' * 66}")

        # ---------- 1. CHON ----------
        if r == 0:
            picked, used = pick_seed_set(pool, a.seed_hours, a.seed)
            print(f"  seed set ngau nhien: {len(picked)} cau = {used:.3f} gio")
        else:
            cands = pool.unlabeled
            if a.candidate_size and len(cands) > a.candidate_size:
                rng = random.Random(a.seed * 1000 + r)   # dung chung moi strategy
                cands = sorted(rng.sample(cands, a.candidate_size))
            print(f"  tap ung vien: {len(cands)} cau")

            ctx = SelectionContext(
                candidates=cands, manifest=train_mf, labeled=pool.labeled,
                model=model, processor=processor,
                round_idx=r, seed=a.seed, budget_hours=a.budget_hours,
            )
            t0 = time.time()
            ranked = strategy.select(ctx)
            sel_sec = time.time() - t0

            picked, used = pool.take_until_budget(ranked, a.budget_hours)
            print(f"  {strategy.name} chon {len(picked)} cau = {used:.3f} gio "
                  f"(xep hang mat {sel_sec:.1f}s)")

        # ---------- 2. GAN NHAN (duong duy nhat) ----------
        oracle.query(picked, round_idx=r)
        pool.add_labeled(picked, round_idx=r)
        oracle.audit(pool)
        print(f"  {pool.summary()}")

        # ---------- 3. HUAN LUYEN LAI TU DAU ----------
        labeled_tx = oracle.labeled_texts()      # chi cau DA query
        run_dir = out_dir / f"round{r}"
        t0 = time.time()
        model, processor = finetune(
            pool.labeled, train_mf, labeled_tx,
            out_dir=run_dir, max_steps=a.max_steps, seed=a.seed,
            per_device_batch=a.batch, grad_accum=a.grad_accum,
            num_workers=a.num_workers,
        )
        train_sec = time.time() - t0

        # ---------- 4. DANH GIA ----------
        t0 = time.time()
        res = evaluate(model, processor, test_ids, test_mf, test_tx,
                       n_examples=3)
        eval_sec = time.time() - t0

        rec = {
            "strategy": strategy.name,
            "seed": a.seed,
            "round": r,
            "labeled_hours": round(pool.labeled_hours, 4),
            "n_labeled": len(pool.labeled),
            "n_picked": len(picked),
            "hours_picked": round(used, 4),
            "wer": round(res["wer"], 2),
            "cer": round(res["cer"], 2),
            "n_empty_hyp": res["n_empty_hyp"],
            "train_sec": round(train_sec, 1),
            "eval_sec": round(eval_sec, 1),
            "round_sec": round(time.time() - t_round, 1),
        }
        records.append(rec)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        pool.save(out_dir / f"pool_round{r}.json")

        print(f"\n  >>> VONG {r}: {pool.labeled_hours:.2f} gio "
              f"-> WER {res['wer']:.2f} %  CER {res['cer']:.2f} %  "
              f"({(time.time() - t_round) / 60:.0f} phut)")

        # Xoa checkpoint vong truoc de khong day Drive (giu vong cuoi)
        if a.keep_last_only and r > 0:
            prev = out_dir / f"round{r - 1}"
            for p in prev.glob("*.safetensors"):
                p.unlink(missing_ok=True)
            for p in prev.glob("*.bin"):
                p.unlink(missing_ok=True)

    # ---------- BANG TONG KET ----------
    # Doc lai TU FILE de bang day du ca cac vong da chay o phien truoc.
    all_recs = []
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            all_recs = [json.loads(l) for l in f if l.strip()]
    all_recs = all_recs or records

    print()
    print("=" * 66)
    print(f"  DUONG CONG HOC — {strategy.name} (seed {a.seed})")
    print("=" * 66)
    print(f"  {'vong':>4} | {'gio nhan':>9} | {'so cau':>7} | {'WER %':>7} | {'CER %':>7}")
    print("-" * 66)
    for rec in all_recs:
        print(f"  {rec['round']:>4} | {rec['labeled_hours']:>9.2f} | "
              f"{rec['n_labeled']:>7} | {rec['wer']:>7.2f} | {rec['cer']:>7.2f}")
    print("=" * 66)
    print(f"  -> {results_path}")
    print(f"  -> {out_dir / 'oracle_log.jsonl'}  (nhat ky gan nhan)")

    with open(out_dir / "oracle_report.json", "w") as f:
        json.dump(oracle.report(), f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--train-transcripts", required=True)
    p.add_argument("--test-manifest", required=True)
    p.add_argument("--test-transcripts", required=True)
    p.add_argument("--strategy", default="random")
    p.add_argument("--rounds", type=int, default=4,
                   help="so vong BO SUNG sau seed set")
    p.add_argument("--seed-hours", type=float, default=1.0)
    p.add_argument("--budget-hours", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=None)
    # GIU NGUYEN batch hieu dung = batch * grad_accum = 16 cau khi doi GPU,
    # neu khong ket qua khong con so sanh duoc voi lan neo o tuan 3.
    #   T4   16GB : --batch 4  --grad-accum 4
    #   L4   24GB : --batch 8  --grad-accum 2
    #   A100 40GB : --batch 16 --grad-accum 1
    p.add_argument("--batch", type=int, default=4,
                   help="so cau moi lan forward tren GPU")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="so lan tich luy gradient truoc khi cap nhat")
    p.add_argument("--num-workers", type=int, default=2,
                   help="luong doc du lieu; 4-8 khi dung A100")
    p.add_argument("--candidate-size", type=int, default=5000,
                   help="0 = dung toan bo tap chua nhan")
    p.add_argument("--test-subset", type=int, default=0,
                   help="0 = danh gia tren toan bo test set")
    p.add_argument("--keep-last-only", action="store_true",
                   help="xoa trong so cua vong truoc de tiet kiem dung luong")
    p.add_argument("--resume", action="store_true",
                   help="chay tiep tu vong cuoi da luu trong --out-dir "
                        "(dung khi phien Colab bi ngat giua duong)")
    p.add_argument("--out-dir", default="runs/al")
    main(p.parse_args())