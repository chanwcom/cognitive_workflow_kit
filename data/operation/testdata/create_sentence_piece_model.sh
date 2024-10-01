#!/bin/bash

# 1. "train-all.trans_without_uttid.txt" was obtained by removing the utterance
#    id part from the original LibriSpeech transcript using the following
#    command.
#
#    cat train-all.trans.txt | perl -pe s'/^\S+\s+//g' > \
#        train-all.trans_without_uttid.txt 
#
# 2. We used the SentencePiece version 0.1.92 to create the model and the
#    vocab.
#
# 3. The vocab file is not directly used in the unit test, but it is included
#    as a reference.

spm_train --input=train-all.trans_without_uttid.txt \
          --model_prefix=model_unigram_256  \
          --vocab_size=256 \
          --character_coverage=1.0 \
          --model_type=unigram
