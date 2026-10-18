"""
RandomStrategy — BASELINE QUAN TRONG NHAT.

TAI SAO RANDOM LA BASELINE NGHIEM TUC, KHONG PHAI "cho co"
----------------------------------------------------------
Trong tai lieu AL, Random thang rat nhieu phuong phap "thong minh":
    - BADGE (2020) bao cao CoreSet va Entropy THUA Random o nhieu cau hinh.
    - Bai chon du lieu cho ASR dua tren transcript du doan: THUA Random.
    - Contrastive sampling: KEM HON Random 22% tuong doi o muc 1 gio.

Ly do: khi so nhan con rat it, model con yeu, nen moi diem "khong chac chan"
do model cham cham deu la nhieu. Random thi it nhat dam bao phan phoi mau
giong phan phoi that cua pool.

=> Neu ClusterMargin-ASR khong thang duoc Random mot cach co y nghia
   thong ke (3 seed), ket luan cua do an la "khong thang", va do VAN LA
   mot ket qua dung bao cao. Dieu khong duoc phep la khong co Random
   de so.

LUU Y VE SEED
-------------
Seed trong moi vong = seed goc + round_idx. Neu dung nguyen seed goc
cho moi vong, thu tu shuffle giong nhau moi vong -> cac cau dau danh sach
luon duoc uu tien, lam lech mau.
"""

import random
from typing import List

from .base import SelectionContext, Strategy


class RandomStrategy(Strategy):
    name = "random"
    needs_model = False

    def select(self, ctx: SelectionContext) -> List[str]:
        ids = list(ctx.candidates)
        random.Random(ctx.seed + ctx.round_idx).shuffle(ids)
        return ids


class LongestFirstStrategy(Strategy):
    """
    Baseline CHAN DOAN, khong phai de thang.

    Dung de do THIEN LECH DO DAI: neu mot chien luoc "thong minh" cho ket qua
    gan voi Longest, nghia la diem do khong chac chan cua no thuc chat chi
    dang do do dai cau (cau dai -> tong entropy lon). Day la bay pho bien
    khi chuyen uncertainty tu phan loai anh sang mo hinh chuoi.
    """
    name = "longest"
    needs_model = False

    def select(self, ctx: SelectionContext) -> List[str]:
        return sorted(ctx.candidates,
                      key=lambda u: -ctx.manifest[u]["duration"])


class ShortestFirstStrategy(Strategy):
    """
    Baseline CHAN DOAN phia doi dien.

    Voi ngan sach tinh bang GIO, chon cau ngan se duoc NHIEU CAU hon trong
    cung so gio -> nhieu cau = nhieu buoc gradient da dang hon. Day la
    baseline ngam ma nhieu bai AL cho ASR quen mat, va no khong he yeu.
    """
    name = "shortest"
    needs_model = False

    def select(self, ctx: SelectionContext) -> List[str]:
        return sorted(ctx.candidates,
                      key=lambda u: ctx.manifest[u]["duration"])