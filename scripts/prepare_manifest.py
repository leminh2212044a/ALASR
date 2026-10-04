"""
Dựng manifest cho LibriSpeech.

THIẾT KẾ QUAN TRỌNG: tách làm HAI file
---------------------------------------
    manifest.jsonl    : utt_id, path, duration, speaker, chapter   (KHÔNG có transcript)
    transcripts.jsonl : utt_id, text

Lý do: từ tuần 4, lớp Oracle sẽ là nơi DUY NHẤT được đọc transcripts.jsonl.
Mọi module khác chỉ thấy manifest. Đây là hàng rào chống label leakage,
dựng ngay từ tuần 3 để không phải sửa lại sau.

CÁCH DÙNG
---------
    python scripts/prepare_manifest.py \
        --librispeech-root /data/LibriSpeech \
        --split train-clean-100 \
        --out-dir data/librispeech

Cấu trúc LibriSpeech gốc:
    LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac
    LibriSpeech/train-clean-100/103/1240/103-1240.trans.txt
"""

import argparse
import json
import os
from pathlib import Path

import soundfile as sf


def read_transcripts(split_dir: Path) -> dict:
    """Đọc toàn bộ file .trans.txt -> dict[utt_id] = text (chữ hoa)."""
    texts = {}
    for trans_file in split_dir.rglob("*.trans.txt"):
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, text = line.partition(" ")
                texts[utt_id] = text
    return texts


def build(librispeech_root: Path, split: str, out_dir: Path):
    split_dir = librispeech_root / split
    if not split_dir.is_dir():
        raise SystemExit(f"Khong thay thu muc: {split_dir}")

    print(f"[1/3] Doc transcript tu {split_dir} ...")
    texts = read_transcripts(split_dir)
    print(f"      -> {len(texts)} transcript")

    print(f"[2/3] Quet file audio va do thoi luong ...")
    manifest_rows, transcript_rows = [], []
    missing_text, total_sec = 0, 0.0

    flacs = sorted(split_dir.rglob("*.flac"))
    for i, path in enumerate(flacs, 1):
        utt_id = path.stem
        if utt_id not in texts:
            missing_text += 1
            continue

        # sf.info chi doc header -> rat nhanh, khong nap toan bo audio
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        total_sec += duration

        speaker, chapter = utt_id.split("-")[0], utt_id.split("-")[1]
        manifest_rows.append({
            "utt_id": utt_id,
            "path": str(path.resolve()),
            "duration": round(duration, 3),
            "speaker": speaker,
            "chapter": chapter,
        })
        transcript_rows.append({"utt_id": utt_id, "text": texts[utt_id]})

        if i % 5000 == 0:
            print(f"      {i}/{len(flacs)} ...")

    print(f"[3/3] Ghi file ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{split}.manifest.jsonl"
    transcript_path = out_dir / f"{split}.transcripts.jsonl"

    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(transcript_path, "w", encoding="utf-8") as f:
        for r in transcript_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Bao cao kiem tra ---
    durs = [r["duration"] for r in manifest_rows]
    durs_sorted = sorted(durs)
    n = len(durs)
    print()
    print("=" * 60)
    print(f"  split           : {split}")
    print(f"  so utterance    : {n}")
    print(f"  tong thoi luong : {total_sec/3600:.2f} gio")
    print(f"  do dai min      : {durs_sorted[0]:.2f} s")
    print(f"  do dai median   : {durs_sorted[n//2]:.2f} s")
    print(f"  do dai max      : {durs_sorted[-1]:.2f} s")
    print(f"  do dai trung binh: {total_sec/n:.2f} s")
    print(f"  so speaker      : {len({r['speaker'] for r in manifest_rows})}")
    if missing_text:
        print(f"  CANH BAO: {missing_text} file audio khong co transcript")
    print("=" * 60)
    print(f"  -> {manifest_path}")
    print(f"  -> {transcript_path}")
    print()
    print("  KIEM TRA: train-clean-100 phai ra ~28539 utterance, ~100.6 gio, 251 speaker")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech-root", required=True, type=Path)
    p.add_argument("--split", default="train-clean-100")
    p.add_argument("--out-dir", default="data/librispeech", type=Path)
    a = p.parse_args()
    build(a.librispeech_root, a.split, a.out_dir)