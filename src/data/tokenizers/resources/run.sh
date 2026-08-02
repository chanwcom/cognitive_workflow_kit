#!/usr/bin/bash
#
# Train SentencePiece tokenizers with multiple vocabulary sizes.

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}" || exit 1

OUTPUT_DIR=/mnt/kioxia_exeria/home/chanwcom/tmp
TOKENIZATION_TYPE=unigram

# List of vocabulary sizes to iterate through.
# 8192 is used as the standard power of 2 for the last value.
readonly VOCAB_SIZES=(32 128 512 2048 8192)

for size in "${VOCAB_SIZES[@]}"; do
  echo "------------------------------------------------"
  echo "Training SentencePiece Model with vocab_size: ${size}"
  echo "------------------------------------------------"

  python src/data/tokenizers/train_spm.py \
    --input "${REPO_ROOT}/src/data/tokenizers/resources/libri_raw.txt" \
    --output_dir "${OUTPUT_DIR}/models/asr" \
    --name "librispeech_${TOKENIZATION_TYPE}_${size}" \
    --vocab_size "${size}" \
    --type $TOKENIZATION_TYPE
done

echo "All training processes have been completed."
