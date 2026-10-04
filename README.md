# ALASR — Active Learning cho ASR

## Cấu trúc

```
alasr/
├── scripts/
│   ├── prepare_manifest.py   # Tuần 3 — dựng manifest + tách transcript
│   ├── anchor_run.py         # Tuần 3 — NEO KẾT QUẢ
│   └── measure_time.py       # Tuần 3 — đo GPU-hour thực tế
├── models/
│   └── w2v2_ctc.py           # fine-tune + evaluate (gọi lại mỗi vòng AL)
├── data/                     # Tuần 4 — Pool, Oracle
├── strategies/               # Tuần 4+ — Random, Margin, ClusterMargin...
├── eval/                     # Tuần 10+ — metrics, phân tích
└── runs/                     # kết quả
```

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Tuần 3 — ba lệnh

```bash
# 1. Dựng manifest cho pool và test set
python scripts/prepare_manifest.py --librispeech-root /data/LibriSpeech \
       --split train-clean-100 --out-dir data/librispeech
python scripts/prepare_manifest.py --librispeech-root /data/LibriSpeech \
       --split test-clean --out-dir data/librispeech
python scripts/prepare_manifest.py --librispeech-root /data/LibriSpeech \
       --split test-other --out-dir data/librispeech

# 2. NEO KẾT QUẢ — kỳ vọng WER 20–30% trên test-clean
python scripts/anchor_run.py \
    --train-manifest    data/librispeech/train-clean-100.manifest.jsonl \
    --train-transcripts data/librispeech/train-clean-100.transcripts.jsonl \
    --test-manifest     data/librispeech/test-clean.manifest.jsonl \
    --test-transcripts  data/librispeech/test-clean.transcripts.jsonl \
    --hours 1.0 --seed 0 --out-dir runs/anchor

# 3. Đo thời gian thực tế → tính lại ngân sách GPU
python scripts/measure_time.py \
    --train-manifest    data/librispeech/train-clean-100.manifest.jsonl \
    --train-transcripts data/librispeech/train-clean-100.transcripts.jsonl \
    --hours 1 5 10
```

## Nguyên tắc thiết kế

| Nguyên tắc | Hiện thực ở đâu |
|---|---|
| Ngân sách đếm bằng **giờ**, không phải số câu | `duration` trong manifest, `sample_by_hours()` |
| Chống **label leakage** | `manifest.jsonl` tách khỏi `transcripts.jsonl` |
| **Neo kết quả** trước khi sáng tạo | `anchor_run.py` |
| Feature encoder **đóng băng** | `model.freeze_feature_encoder()` |
| Vocab **ký tự** (~32 token) | `build_vocab()` — giữ gradient embedding ở 23k chiều |
