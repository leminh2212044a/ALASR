"""
Giao dien chung cho moi chien luoc chon du lieu.

HOP DONG
--------
    ranked = strategy.select(ctx)

`ranked` la DANH SACH utt_id SAP XEP THEO DO UU TIEN GIAM DAN.
Chien luoc KHONG tu quyet dinh lay bao nhieu cau — viec cat theo
ngan sach gio do Pool.take_until_budget() lam. Nho the:

    - moi chien luoc chiu dung mot ngan sach gio nhu nhau
    - khong chien luoc nao "gian" bang cach chon cau dai

NHUNG GI CHIEN LUOC DUOC PHEP THAY (xem SelectionContext)
---------------------------------------------------------
    candidates : utt_id chua co nhan  (da duoc lay mau con, dung chung
                 cho moi chien luoc trong cung vong -> so sanh cong bang)
    manifest   : path, duration, speaker            (KHONG co text)
    model      : model vua fine-tune o vong truoc
    processor  : de tinh posterior / embedding
    labeled    : utt_id DA co nhan — chi ID, KHONG kem transcript

KHONG duoc phep: Oracle, transcripts, hay text cua cau chua gan nhan.
Day la ly do `select()` nhan ctx thay vi nhan truc tiep cac tham so —
de khong ai vo tinh truyen transcripts vao.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SelectionContext:
    candidates: List[str]
    manifest: Dict[str, dict]
    labeled: List[str] = field(default_factory=list)
    model: Optional[object] = None
    processor: Optional[object] = None
    round_idx: int = 0
    seed: int = 0
    budget_hours: float = 1.0

    def __post_init__(self):
        for r in self.manifest.values():
            if "text" in r or "transcript" in r:
                raise ValueError("manifest co transcript -> RO RI NHAN")


class Strategy(ABC):
    name = "base"

    #: True neu chien luoc can model de tinh diem (uncertainty, embedding...).
    #: Random / Longest / Shortest khong can -> vong 1 khoi phai cho train.
    needs_model = False

    @abstractmethod
    def select(self, ctx: SelectionContext) -> List[str]:
        """Tra ve utt_id sap xep theo do uu tien giam dan."""
        raise NotImplementedError

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"