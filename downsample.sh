#!/bin/sh
ffmpeg -i data/train.mp4 -vf scale=128:128 data/train_128.mp4
ffmpeg -i data/test.mp4 -vf scale=128:128 data/test_128.mp4
