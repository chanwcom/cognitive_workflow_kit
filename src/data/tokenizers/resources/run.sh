python src/data/tokenizers/train_spm.py \
    --input libri_raw.txt \
    --output_dir models/asr \
    --name librispeech_bpe \
    --vocab_size 1024 \
    --type bpe
