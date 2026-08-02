#!/bin/bash
#

LIBRI_LIGHT_TOP="/mnt/nas2dual/database/libri_light_finetuning/org_decompressed"
LIBRI_LIGHT_OUT="/mnt/nas2dual/database/libri_light_finetuning_webdataset"
LIBRI_SPEECH_TOP="/mnt/nas2dual/database/libri_speech/org_decompressed"
LIBRI_SPEECH_OUT="/mnt/nas2dual/database/libri_speech_webdataset_new_oct_2025"

#  --filter-list "$RESOURCE_DIR/aihub_consulting_filtered_id.txt" \
#--filter-list "$RESOURCE_DIR/aihub_consulting_filtered_id.txt" \

# LibriLight 1-hr fine-tuning set
python create_librispeech_webdataset.py \
    --dataset_dir "$LIBRI_LIGHT_TOP/1h" \
    --output_dir "$LIBRI_LIGHT_OUT/1h" \
    --shard_size_gb 5.0

# LibriLight 10-hr fine-tuning set
python create_librispeech_webdataset.py \
    --dataset_dir $LIBRI_LIGHT_TOP  \
    --output_dir "$LIBRI_LIGHT_OUT/10h" \
    --shard_size_gb 5.0

for split in test-clean test-other; do
     python create_librispeech_webdataset.py \
         --dataset_dir $LIBRI_SPEECH_TOP/$split \
         --output_dir $LIBRI_SPEECH_OUT/$split \
         --shard_size_gb 5.0
done


# LibriTrain test-clean, test-other
for split in train-clean-100 train-clean-360 train-other-500; do
     python create_librispeech_webdataset.py \
         --dataset_dir $LIBRI_SPEECH_TOP/$split \
         --output_dir $LIBRI_SPEECH_OUT/$split \
         --shard_size_gb 5.0
done
