"""
Oracle — DUONG DUY NHAT di tu du lieu chua nhan sang du lieu co nhan.

VAN DE: RO RI NHAN (label leakage)
----------------------------------
Trong Active Learning, gia dinh co ban la: chien luoc chon KHONG biet
transcript cua cau no dang chon. Neu bat ky doan code nao doc transcript
cua cau chua gan nhan — du chi de tinh do dai cau, dem tu, hay loc cau
"xau" — thi toan bo ket qua thi nghiem MAT GIA TRI, vi trong thuc te
transcript do chua ton tai.

Day la loi pho bien nhat trong cac bai AL cho ASR, va cung la thu hoi
dong dau tien cua phan phan bien.

CACH CHONG
----------
1. prepare_manifest.py tach lam HAI file:
       manifest.jsonl     -> path, duration, speaker   (chien luoc duoc doc)
       transcripts.jsonl  -> text                      (CHI Oracle duoc doc)
2. Chi lop nay nap transcripts.jsonl. Strategy khong duoc nhan Oracle.
3. Moi lan query duoc GHI LOG kem vong va dau thoi gian.
4. audit() kiem tra: moi cau da bi query deu nam trong tap co nhan
   cua Pool, va khong co cau nao bi query nhieu lan.

CACH DUNG
---------
    oracle = Oracle("data/librispeech/train-clean-100.transcripts.jsonl",
                    log_path="runs/al_random/oracle_log.jsonl")
    texts = oracle.query(picked_ids, round_idx=1)
    pool.add_labeled(picked_ids, round_idx=1)
    oracle.audit(pool)
"""

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List


class Oracle:
    def __init__(self, transcripts_path, log_path=None):
        self.__transcripts: Dict[str, str] = {}
        with open(transcripts_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.__transcripts[r["utt_id"]] = r["text"]

        self._queried: Dict[str, int] = {}      # utt_id -> vong
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

        print(f"Oracle: nap {len(self.__transcripts)} transcript "
              f"(tap nay KHONG duoc doc o bat ky noi khac)")

    # ---------------- gan nhan ----------------

    def query(self, uids: Iterable[str], round_idx: int) -> Dict[str, str]:
        """
        'Thue nguoi gan nhan' cho cac cau nay. Tra ve transcript.
        Moi cau chi duoc query MOT LAN trong ca lan chay.
        """
        uids = list(uids)
        out = {}
        for u in uids:
            if u in self._queried:
                raise ValueError(
                    f"{u} da duoc query o vong {self._queried[u]} -> "
                    f"dang tinh trung ngan sach gan nhan.")
            if u not in self.__transcripts:
                raise KeyError(f"khong co transcript cho {u}")
            self._queried[u] = round_idx
            out[u] = self.__transcripts[u]

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for u in uids:
                    f.write(json.dumps({"round": round_idx, "utt_id": u,
                                        "ts": time.time()}) + "\n")
        return out

    def labeled_texts(self) -> Dict[str, str]:
        """Transcript cua NHUNG CAU DA DUOC QUERY — dung de fine-tune."""
        return {u: self.__transcripts[u] for u in self._queried}

    @property
    def n_queried(self) -> int:
        return len(self._queried)

    # ---------------- kiem toan ----------------

    def audit(self, pool) -> None:
        """
        Kiem tra tinh toan ven. Nem AssertionError neu phat hien bat thuong.
        Goi SAU MOI VONG.
        """
        q = set(self._queried)
        lab = set(pool.labeled)

        assert q == lab, (
            f"KHONG KHOP: Oracle da query {len(q)} cau nhung Pool co "
            f"{len(lab)} cau co nhan.\n"
            f"  query nhung chua vao Pool: {sorted(q - lab)[:5]}\n"
            f"  o Pool nhung chua query  : {sorted(lab - q)[:5]}\n"
            f"  -> neu co cau o Pool ma chua query, nghia la co doan code "
            f"lay transcript khong qua Oracle = RO RI NHAN."
        )
        assert len(self._queried) == len(set(self._queried)), "query trung"
        print(f"    audit OK: {len(q)} cau, khong ro ri nhan")

    def report(self) -> List[dict]:
        """Thong ke so cau duoc query o tung vong — de ve bang trong bao cao."""
        per_round: Dict[int, int] = {}
        for r in self._queried.values():
            per_round[r] = per_round.get(r, 0) + 1
        return [{"round": r, "n_queried": n} for r, n in sorted(per_round.items())]