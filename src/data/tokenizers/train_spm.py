"""Generic script to train SentencePiece tokenizer from text files."""

import argparse
import os
import sentencepiece as spm


def train_tokenizer(input_file, model_prefix, vocab_size, model_type):
    """Trains a SentencePiece model with specified parameters.

    Args:
        input_file: Path to the plain text file for training.
        model_prefix: Output filename prefix for .model and .vocab.
        vocab_size: Number of tokens in the vocabulary.
        model_type: Algorithm type (unigram, bpe, char, or word).
    """
    cmd = (
        f"--input={input_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --model_type={model_type} "
        f"--character_coverage=1.0 --pad_id=0 --eos_id=1 "
        f"--bos_id=2 --unk_id=3"
    )
    spm.SentencePieceTrainer.train(cmd)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input text file path"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models", help="Output directory"
    )
    parser.add_argument(
        "--name", type=str, default="tokenizer", help="Model name prefix"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=5000, help="Vocabulary size"
    )
    parser.add_argument(
        "--type", type=str, default="unigram", help="unigram, bpe, char, word"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    prefix = os.path.join(args.output_dir, args.name)
    
    print(f"Training {args.type} tokenizer: {prefix}...")
    train_tokenizer(args.input, prefix, args.vocab_size, args.type)
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
