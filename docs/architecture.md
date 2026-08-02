# Architecture

This document describes the design principles behind
`cognitive_workflow_kit` (cwk) and explains how the core modules fit
together.

## Design Principles

1. **Separation of building blocks and assembled artifacts.**
   `layer/` contains reusable, low-level building blocks (attention,
   convolution, normalization, etc.). `model/` contains complete models
   assembled from those layers. A layer should never depend on a model,
   only the reverse.

2. **Framework independence at the interface level.**
   Where practical, `tensorflow/` and `pytorch/` implementations live
   side by side under the same module (e.g. `layer/tensorflow`,
   `layer/pytorch`) and expose a comparable interface, so that switching
   frameworks does not require relearning the toolkit's API.

3. **Configuration over code changes.**
   Users should be able to assemble new model/loss/optimizer combinations
   by editing a config file, not by editing library code. This is the
   responsibility of `registry/`.

4. **Training and evaluation are first-class, not scripts.**
   Rather than duplicating the training loop across scripts (as in
   `scripts/train/*.py`), the loop itself lives in `cwk/train/trainer.py`
   and is extended through callbacks, not copy-paste.

5. **Library code vs. entry-point code are physically separated.**
   `cwk/` is importable library code with unit tests next to each source
   file. `scripts/` contains thin, non-reusable entry points that wire
   library components together for a specific job.

## Module Overview

```
cwk/
├── core/        Common base classes shared across modules
├── data/        Data loading, formats (proto schemas), and dataset ops
├── math/        Signal processing and statistical utilities
├── layer/       Reusable neural network building blocks
├── model/       Complete models assembled from layers
├── loss/        Loss functions
├── metric/      Evaluation metrics (WER, accuracy, F1, ...)
├── optimizer/   Optimizer and learning-rate scheduler wrappers
├── registry/    Config-driven factory for assembling components
└── train/       Trainer, evaluator, checkpointing, and callbacks
```

### `data/`

Responsible for everything between "raw file on disk" and "batched tensor
ready for the model."

- `format/` — protobuf schema definitions describing on-disk data formats
- `source/` — code that reads raw sources (audio files, text corpora)
- `ops/` — `tf.data` / `torch.utils.data` operations, tokenization/codec
  logic

### `layer/` and `model/`

`layer/` provides composable pieces (e.g. `AttentionLayer`,
`ConformerBlockLayer`, `SubsamplingLayer`). `model/` combines these into
end-to-end architectures (e.g. `Wav2VecModel`, `BertModel`). A model
should be describable largely as a composition of existing layers plus a
small amount of glue code.

### `loss/` and `metric/`

These are kept separate on purpose: a **loss** is what the optimizer
minimizes (must be differentiable), while a **metric** is what a human
reads to judge model quality (e.g. Word Error Rate) and need not be
differentiable. Conflating the two tends to make loss functions harder to
reuse for evaluation-only purposes.

### `registry/`

A lightweight factory layer that maps string identifiers in a config file
to concrete classes in `layer/`, `model/`, `loss/`, and `optimizer/`. This
is what allows a user to change models by editing a YAML/JSON config
instead of Python code. New components are added to the toolkit by
registering them here, not by modifying existing training code.

### `train/`

Encapsulates the training and evaluation loop:

- `trainer.py` — the core training loop (forward/backward pass, gradient
  updates, calling callbacks at defined points)
- `evaluator.py` — runs a model over an evaluation set and computes
  `metric/` values
- `checkpoint/` — saving/restoring model and optimizer state
- `callback/` — extension points invoked by the trainer, e.g.:
  - `eval_callback.py` — triggers `evaluator.py` on a schedule
  - `checkpoint_callback.py` — saves checkpoints, e.g. on best-metric
  - `logging_callback.py` — writes logs/TensorBoard summaries
  - `early_stopping_callback.py` — halts training based on eval results

Callbacks exist so that evaluation, checkpointing, and logging policy
(*when* and *how often*) are decoupled from the training loop itself
(*how* a single step is executed).

## Data Flow

```
        ┌─────────┐      ┌────────┐      ┌─────────┐
raw --> │  data/  │ ---> │ model/ │ ---> │  loss/  │ --> gradients
        └─────────┘      └────────┘      └─────────┘
                              │
                              v
                         ┌─────────┐      ┌──────────┐
                         │ metric/ │ <--- │ train/   │ (evaluator)
                         └─────────┘      │ trainer  │
                                           └──────────┘
                                                │
                                    callbacks: checkpoint / log / early-stop
```

## Extending the Toolkit

| I want to... | Add code under... |
|---|---|
| Add a new reusable neural network component | `cwk/layer/<framework>/` |
| Add a new end-to-end model | `cwk/model/<framework>/` |
| Add a new dataset format or loader | `cwk/data/` |
| Add a new evaluation metric | `cwk/metric/` |
| Change what happens during/after training steps | `cwk/train/callback/` |
| Make a new component config-selectable | Register it in `cwk/registry/` |
| Add a runnable script for a new task | `scripts/<category>/` |
| Add a learning example for new users | `examples/` |

## Non-Goals

- `cwk/` does not aim to support every deep learning framework — only
  TensorFlow and PyTorch, and only where the maintainers actively use
  both.
- Experimental or one-off scratch scripts are intentionally kept out of
  the repository (or isolated to a clearly named sandbox location) to
  keep the core library trustworthy and easy to navigate.
