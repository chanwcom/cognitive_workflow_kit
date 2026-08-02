# Getting Started

This guide walks you through installing `cognitive_workflow_kit` (cwk) and
running your first training job.

## Prerequisites

- [Bazel](https://bazel.build/) (see `MODULE.bazel` for the required version)
- Python 3.x
- TensorFlow and/or PyTorch, depending on which backend you plan to use
- (Optional) A CUDA-capable GPU for accelerated training

## Installation

Clone the repository:

```bash
git clone https://github.com/<org>/cognitive_workflow_kit.git
cd cognitive_workflow_kit
```

Build the project with Bazel:

```bash
./bazel_wrapper.sh build //...
```

Run the test suite to confirm your environment is set up correctly:

```bash
./bazel_wrapper.sh test //...
```

## Repository Layout

A quick orientation before you dive in:

| Path | Purpose |
|---|---|
| `cwk/` | Core library: data, model, layer, loss, training utilities |
| `scripts/` | Production entry points for training, inference, and data prep |
| `examples/` | Minimal, runnable examples to learn the toolkit |
| `configs/` | Experiment configuration files |
| `docs/` | Documentation (you are here) |

See [`architecture.md`](architecture.md) for a deeper explanation of how
these pieces fit together.

## Your First Training Run

The fastest way to get a feel for the toolkit is to run one of the examples:

```bash
./bazel_wrapper.sh run //examples:train_wav2vec
```

This will:

1. Load a small sample dataset via `cwk/data`
2. Build a model from `cwk/model` using a config in `configs/`
3. Run a short training loop via `cwk/train/trainer.py`
4. Report evaluation metrics via `cwk/metric`

## Training on Your Own Data

1. **Prepare your data.** Convert your raw dataset into the toolkit's
   expected format using a script under `scripts/data_prep/`, e.g.:

   ```bash
   ./bazel_wrapper.sh run //scripts/data_prep:convert_to_tfrecord -- \
     --input_dir=/path/to/raw/data \
     --output_dir=/path/to/output
   ```

2. **Write a config.** Copy an existing config from `configs/` and adjust
   the model, optimizer, and training parameters to your needs.

3. **Launch training.** Use one of the scripts under `scripts/train/`, or
   write your own driver script using `cwk.train.trainer.Trainer`.

4. **Monitor training.** Attach callbacks under `cwk/train/callback/` for
   logging, checkpointing, and early stopping.

## Next Steps

- Read [`architecture.md`](architecture.md) to understand the design
  principles behind the toolkit.
- Browse `examples/` for more end-to-end workflows.
- See `CONTRIBUTING.md` if you'd like to contribute a new model, layer, or
  dataset loader.

## Getting Help

If you run into issues, please open a GitHub issue with:

- The command you ran
- The full error message
- Your environment (OS, Python version, Bazel version, GPU/CPU)
