#!/bin/bash
LIBRI_SPEECH_TEXT="/mnt/nas2dual/database/libri_speech_text"
LIBRI_SPEECH_TEXT_OUT="/mnt/nas2dual/database/libri_speech_text_webdataset"

python ../create_librispeech_text_webdataset.py \
    --input_file "$LIBRI_SPEECH_TEXT/out.txt" \
    --output_dir "$LIBRI_SPEECH_TEXT_OUT" \
    --shard_size_gb 5.0
