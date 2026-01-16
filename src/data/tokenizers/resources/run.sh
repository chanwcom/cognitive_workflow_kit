죄송합니다. 구글 스타일 가이드와 80자 제한을 엄격하게 고려하지 못했네요. 요청하신 대로 Google Shell Style Guide를 준수하여, 한 줄에 80자가 넘지 않도록 백슬래시(\)로 줄바꿈을 처리하고 주석 형식을 수정한 버전입니다.

수정된 스크립트
Bash

#!/usr/bin/bash
#
# Train SentencePiece tokenizers with multiple vocabulary sizes.

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}" || exit 1

# List of vocabulary sizes to iterate through.
# 8192 is used as the standard power of 2 for the last value.
readonly VOCAB_SIZES=(32 128 512 2048 8192)

for size in "${VOCAB_SIZES[@]}"; do
  echo "------------------------------------------------"
  echo "Training SentencePiece Model with vocab_size: ${size}"
  echo "------------------------------------------------"

  python src/data/tokenizers/train_spm.py \
    --input "${REPO_ROOT}/src/data/tokenizers/resources/libri_raw.txt" \
    --output_dir "models/asr" \
    --name "librispeech_bpe_${size}" \
    --vocab_size "${size}" \
    --type "bpe"
done

echo "All training processes have been completed."
