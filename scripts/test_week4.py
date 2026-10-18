"""
Kiem tra Pool + Oracle KHONG CAN GPU, khong can tai LibriSpeech.
Chay trong ~1 giay. Chay truoc khi dot GPU cho run_al.py.

    python scripts/test_week4.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.oracle import Oracle
from data.pool import Pool
from strategies import build
from strategies.base import SelectionContext


def fake_data(n=100):
    """n cau gia, do dai 2..20 giay."""
    mf = {f"u{i:03d}": {"utt_id": f"u{i:03d}", "path": f"/fake/u{i}.flac",
                        "duration": 2.0 + (i % 19), "speaker": f"spk{i % 7}"}
          for i in range(n)}
    tx = {u: f"CAU SO {u}" for u in mf}
    return mf, tx


def ok(msg):
    print(f"  [OK] {msg}")


def main():
    mf, tx = fake_data(300)
    tmp = Path(tempfile.mkdtemp())
    tx_path = tmp / "transcripts.jsonl"
    with open(tx_path, "w", encoding="utf-8") as f:
        for u, t in tx.items():
            f.write(json.dumps({"utt_id": u, "text": t}) + "\n")

    print("\n--- 1. Pool: ngan sach tinh bang GIO ---")
    pool = Pool(mf)
    total = pool.unlabeled_hours
    assert len(pool.unlabeled) == 300
    ok(f"pool co 300 cau = {total:.3f} gio")

    print("\n--- 2. take_until_budget KHONG BAO GIO vuot ngan sach ---")
    for budget in [0.05, 0.1, 0.25, 0.5]:
        picked, used = pool.take_until_budget(pool.unlabeled, budget)
        assert used <= budget + 1e-9, f"vuot: {used} > {budget}"
        assert used > budget * 0.9, f"lay qua it: {used} / {budget}"
        ok(f"ngan sach {budget} gio -> lay {len(picked)} cau = {used:.4f} gio")

    print("\n--- 3. Chien luoc giu dung thu tu uu tien ---")
    ctx = SelectionContext(candidates=pool.unlabeled, manifest=mf,
                           round_idx=1, seed=0, budget_hours=0.1)
    longest = build("longest").select(ctx)
    durs = [mf[u]["duration"] for u in longest]
    assert durs == sorted(durs, reverse=True)
    p_long, _ = pool.take_until_budget(longest, 0.1)
    p_short, _ = pool.take_until_budget(build("shortest").select(ctx), 0.1)
    assert len(p_short) > len(p_long), "shortest phai duoc NHIEU cau hon"
    ok(f"cung 0.1 gio: longest={len(p_long)} cau, shortest={len(p_short)} cau "
       f"-> ngan sach gio da trung hoa thien lech do dai")

    print("\n--- 4. Oracle: duong duy nhat tien tai transcript ---")
    oracle = Oracle(tx_path, log_path=tmp / "log.jsonl")
    picked, used = pool.take_until_budget(pool.unlabeled, 0.1)
    texts = oracle.query(picked, round_idx=0)
    pool.add_labeled(picked, round_idx=0)
    oracle.audit(pool)
    assert set(texts) == set(picked)
    ok(f"query {len(picked)} cau, audit thong qua")

    print("\n--- 5. Oracle bat query TRUNG (tinh trung ngan sach) ---")
    try:
        oracle.query(picked[:1], round_idx=1)
        raise AssertionError("LE RA PHAI BAO LOI")
    except ValueError as e:
        ok(f"bat duoc: {str(e)[:60]}...")

    print("\n--- 6. audit() PHAT HIEN ro ri nhan ---")
    sneaky = pool.unlabeled[0]
    pool.add_labeled([sneaky], round_idx=1)      # vao Pool ma KHONG qua Oracle
    try:
        oracle.audit(pool)
        raise AssertionError("LE RA PHAI BAO LOI")
    except AssertionError as e:
        if "LE RA" in str(e):
            raise
        ok("audit bat duoc cau vao tap co nhan ma khong qua Oracle")

    print("\n--- 7. Pool tu choi manifest co transcript ---")
    bad = {u: {**r, "text": "XIN CHAO"} for u, r in mf.items()}
    try:
        Pool(bad)
        raise AssertionError("LE RA PHAI BAO LOI")
    except ValueError:
        ok("Pool chan manifest lan transcript")

    print("\n--- 8. Random tai lap duoc, moi vong cho thu tu KHAC ---")
    c1 = SelectionContext(candidates=pool.unlabeled, manifest=mf, round_idx=1, seed=0)
    c2 = SelectionContext(candidates=pool.unlabeled, manifest=mf, round_idx=2, seed=0)
    r1a = build("random").select(c1)
    r1b = build("random").select(c1)
    r2 = build("random").select(c2)
    assert r1a == r1b, "cung vong, cung seed -> phai giong nhau"
    assert r1a != r2, "khac vong -> phai khac nhau"
    ok("cung (seed, vong) -> giong; khac vong -> khac")

    print("\n--- 9. Luu / nap trang thai (Colab het phien) ---")
    pool.save(tmp / "pool.json")
    pool2 = Pool(mf)
    pool2.load(tmp / "pool.json")
    assert pool2.labeled == pool.labeled
    assert abs(pool2.labeled_hours - pool.labeled_hours) < 1e-9
    ok("nap lai dung trang thai")

    print("\n--- 10. Chay tiep: Oracle.restore + audit van thong qua ---")
    pool3 = Pool(mf)
    pool3.load(tmp / "pool.json")
    oracle3 = Oracle(tx_path, log_path=tmp / "log.jsonl", append=True)
    oracle3.restore(pool3.round_map)
    oracle3.audit(pool3)
    assert oracle3.n_queried == len(pool3.labeled)
    # sau khi phuc hoi, query tiep mot cau MOI phai duoc chap nhan
    nxt, _ = pool3.take_until_budget(pool3.unlabeled, 0.02)
    oracle3.query(nxt, round_idx=2)
    pool3.add_labeled(nxt, round_idx=2)
    oracle3.audit(pool3)
    ok(f"phuc hoi {len(pool3.labeled) - len(nxt)} cau, query tiep {len(nxt)} cau")

    print("\n" + "=" * 54)
    print("  TAT CA KIEM TRA TUAN 4 DAT. San sang chay run_al.py")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    main()