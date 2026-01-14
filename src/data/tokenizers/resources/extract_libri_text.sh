#!/bin/bash

# Extract text from LibriSpeech .trans.txt files while stripping IDs.
# Usage: ./extract_libri_text.sh <data_dir> <output_file>

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <data_dir> <output_file>"
  exit 1
fi

DATA_DIR=$1
OUTPUT_FILE=$2

# Find files and extract text. Google Style: wrap lines for readability.
find "$DATA_DIR" -name "*.trans.txt" \
    -exec cut -d' ' -f2- {} + > "$OUTPUT_FILE"
