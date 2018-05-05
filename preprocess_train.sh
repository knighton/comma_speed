#/bin/sh
python3 preprocess_train.py \
    --in data/train_128.mp4 \
    --out_train_clips data/train_clips.npy \
    --out_train_indices data/train_indices.npy \
    --out_val_clips data/val_clips.npy \
    --out_val_indices data/val_indices.npy
