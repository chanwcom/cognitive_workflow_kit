# LibriSpeech Text Extraction

This repository contains utility scripts to process the LibriSpeech dataset for ASR (Automatic Speech Recognition) tasks, such as building Trie-based biasing dictionaries or training Language Models (LM).

## Scripts

### extract_libri_text.sh

This script recursively searches for LibriSpeech transcription files (*.trans.txt), removes the utterance IDs from each line, and aggregates the raw text into a single output file.

#### Usage

```bash
./extract_libri_text.sh <data_dir> <output_file>
```

#### Example

Based on the specific environment (NAS storage), use the following command:

```bash
./extract_libri_text.sh \
    /mnt/nas2dual/database/libri_speech/org_decompressed \
    libri_raw.txt
```

## Implementation Standards

* Google Shell Style Guide: The script follows Google's style recommendations, including an 80-character line limit and clear English documentation.
* Efficient Processing: Uses find and cut to handle large-scale dataset directories efficiently without loading entire files into memory.

## Project Structure

* extract_libri_text.sh: The main extraction utility.
* README.md: Documentation for data preparation.

## Data Management

> [!IMPORTANT]
> The output text files (e.g., libri_raw.txt) can be very large. Do not commit these files to the repository.

Ensure your .gitignore includes the following:

```text
# Ignore raw data outputs
libri_raw.txt
*.raw.txt
```
