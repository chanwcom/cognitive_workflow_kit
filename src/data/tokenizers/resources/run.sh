#!/usr/bin/bash

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python src/data/tokenizers/train_spm.py \
    --input $REPO_ROOT/src/data/tokenizers/resources/libri_raw.txt \
    --output_dir models/asr \
    --name librispeech_bpe \
    --vocab_size 1024 \
    --type bpe
