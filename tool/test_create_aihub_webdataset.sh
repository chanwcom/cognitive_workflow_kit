#!/bin/bash

DB_ROOT="/mnt/kioxia_exeria/speech_database/aihub_consulting_datasetkey_100"
DATA_ROOT="$DB_ROOT/012.상담_음성_데이터_decomprssed"
AUDIO_ROOT="$DATA_ROOT/01.데이터/1.Training/원천데이터_1129_add"
LABEL_ROOT="$DATA_ROOT/01.데이터/1.Training/라벨링데이터_1129_add"
RESOURCE_DIR="$DB_ROOT/resource"
#  --filter-list "$RESOURCE_DIR/debug.txt" \

python ./create_aihub_webdataset.py \
  --filter-list "$RESOURCE_DIR/aihub_consulting_filtered_id.txt" \
  --audio-root $AUDIO_ROOT  \
  --label-root $LABEL_ROOT  \
  --output-dir "$DB_ROOT/webdataset" \
  --shard-size-gb 5.0
