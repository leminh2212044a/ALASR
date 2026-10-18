"""
Pool — quan ly tap DA CO NHAN / CHUA CO NHAN, dem ngan sach bang GIO.

TAI SAO DEM BANG GIO MA KHONG DEM BANG SO CAU
---------------------------------------------
Chi phi gan nhan loi noi ti le voi DO DAI AUDIO, khong phai so cau.
Neu dem bang so cau, mot chien luoc chi can uu tien chon cau DAI la
tu dong duoc nhieu du lieu hon -> so sanh mat y nghia.

Toan bo ngan sach trong du an nay deu tinh bang gio:
    - seed set        : 1.0 gio
    - moi vong AL     : +1.0 gio
    - 8 vong          : 1 -> 8 gio

LOP NAY KHONG BAO GIO DOC TRANSCRIPT. Chi Oracle duoc doc.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


class Pool:
    def __init__(self, manifest: Dict[str, dict]):
        """manifest: dict[utt_id] -> {path, duration, speaker, ...}  (KHONG co text)"""
        if any("text" in r or "transcript" in r for r in manifest.values()):
            raise ValueError(
                "manifest chua truong 'text'/'transcript' -> RO RI NHAN. "
                "Dung prepare_manifest.py de tach transcript ra file rieng."
            )
        self.manifest = manifest
        self._labeled: List[str] = []          # giu THU TU duoc chon (de truy vet)
        self._round_of: Dict[str, int] = {}    # utt_id -> vong nao chon no
        self._unlabeled = set(manifest.keys())

    # ---------------- truy van ----------------

    @property
    def labeled(self) -> List[str]:
        return list(self._labeled)

    @property
    def unlabeled(self) -> List[str]:
        """Tra ve list da sap xep -> tai lap duoc giua cac lan chay."""
        return sorted(self._unlabeled)

    def duration(self, uid: str) -> float:
        """Do dai mot cau, tinh bang GIAY."""
        return self.manifest[uid]["duration"]

    def hours(self, uids: Iterable[str]) -> float:
        return sum(self.manifest[u]["duration"] for u in uids) / 3600

    @property
    def labeled_hours(self) -> float:
        return self.hours(self._labeled)

    @property
    def unlabeled_hours(self) -> float:
        return self.hours(self._unlabeled)

    def round_of(self, uid: str) -> int:
        return self._round_of[uid]

    @property
    def round_map(self) -> Dict[str, int]:
        """utt_id -> vong da chon no. Dung de phuc hoi Oracle khi chay tiep."""
        return dict(self._round_of)

    # ---------------- cap nhat ----------------

    def add_labeled(self, uids: Sequence[str], round_idx: int) -> None:
        for u in uids:
            if u not in self._unlabeled:
                raise KeyError(f"{u} khong con trong tap chua nhan (da chon roi?)")
            self._unlabeled.discard(u)
            self._labeled.append(u)
            self._round_of[u] = round_idx

    # ---------------- lay theo ngan sach ----------------

    def take_until_budget(self, ranked: Sequence[str], budget_hours: float,
                          tolerance_frac: float = 0.01) -> Tuple[List[str], float]:
        """
        Lay cau theo DUNG THU TU UU TIEN `ranked` cho toi khi het ngan sach.

        - Giu nguyen thu tu: cau xep hang 1 duoc uu tien tuyet doi.
        - Cau nao lam VUOT ngan sach thi BO QUA, di tiep -> khong bao gio
          vuot qua budget_hours.
        - Dung khi da lap day >= (1 - tolerance_frac) ngan sach.
          Dung ty le (khong dung so giay tuyet doi) de quy tac giong nhau
          du ngan sach la 1 gio hay 0.05 gio.

        Tra ve (danh sach chon, so gio thuc te).
        """
        budget_sec = budget_hours * 3600
        tol_sec = budget_sec * tolerance_frac
        picked, used = [], 0.0

        for uid in ranked:
            if uid not in self._unlabeled:
                continue                            # da co nhan -> bo qua
            d = self.manifest[uid]["duration"]
            if used + d > budget_sec:
                continue                            # qua dai -> thu cau sau
            picked.append(uid)
            used += d
            if budget_sec - used < tol_sec:
                break

        return picked, used / 3600

    # ---------------- luu / nap trang thai ----------------

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "labeled": self._labeled,
                "round_of": self._round_of,
                "labeled_hours": round(self.labeled_hours, 4),
            }, f, indent=2)

    def load(self, path) -> None:
        """Nap lai trang thai -> tiep tuc mot lan chay bi ngat (Colab het phien)."""
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        self._labeled = s["labeled"]
        self._round_of = {k: int(v) for k, v in s["round_of"].items()}
        self._unlabeled = set(self.manifest.keys()) - set(self._labeled)

    def summary(self) -> str:
        return (f"Pool: {len(self._labeled)} cau / {self.labeled_hours:.2f} gio CO NHAN  |  "
                f"{len(self._unlabeled)} cau / {self.unlabeled_hours:.1f} gio CHUA NHAN")