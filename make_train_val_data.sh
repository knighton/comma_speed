#/bin/sh
python3 make_train_val_data.py \
    --in data/train_128.mp4 \
    --out_train_frames data/train_128_train_frames.npy \
    --out_train_indices data/train_128_train_indices.npy \
    --out_val_frames data/train_128_val_frames.npy \
    --out_val_indices data/train_128_val_indices.npy
