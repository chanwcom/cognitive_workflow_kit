cognitive_workflow_kit/
├── README.md                        # Project overview and quick start
├── LICENSE                          # Open source license (required)
├── CONTRIBUTING.md                  # Contribution guidelines and code style
├── CODE_OF_CONDUCT.md               # Community conduct guidelines
├── CHANGELOG.md                     # Version history
├── MODULE.bazel
├── MODULE.bazel.lock
├── WORKSPACE
├── bazel_wrapper.sh
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                   # Build and test automation
│   │   └── lint.yml                 # Style/lint checks
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docs/
│   ├── getting_started.md
│   ├── architecture.md              # System design, diagrams (class_dia, data flow)
│   ├── api/                         # API reference (auto-generated)
│   └── tutorials/
│
├── examples/                        # Minimal, runnable end-to-end examples for external users
│   ├── train_wav2vec.md
│   └── train_bert.md
│
├── third_party/                     # External dependency BUILD files
│   ├── BUILD
│   ├── librosa.BUILD
│   ├── pcre.BUILD
│   ├── soundfile.BUILD
│   └── swig.BUILD
│
├── cwk/                              # Core library package
│   ├── core/                        # Common base classes (operation.py etc.)
│   │
│   ├── data/
│   │   ├── format/                  # Proto schema definitions
│   │   ├── ops/                     # Dataset ops, text codec
│   │   └── source/                  # Raw data loading (audio.py etc.)
│   │       └── testdata/
│   │
│   ├── math/
│   │   ├── signal/                  # STFT, mel-spectrogram, signal processing
│   │   └── stats/                   # Normalization, distribution utilities
│   │
│   ├── layer/
│   │   ├── tensorflow/
│   │   └── pytorch/
│   │
│   ├── model/                       # Assembled models built from layers
│   │   ├── tensorflow/
│   │   └── pytorch/
│   │
│   ├── loss/
│   │   ├── tensorflow/
│   │   └── pytorch/
│   │
│   ├── metric/                      # WER, accuracy, F1, etc.
│   │
│   ├── optimizer/                   # Optimizer and LR scheduler wrappers
│   │
│   ├── registry/                    # Config-driven factory for model/layer/loss assembly
│   │
│   └── train/
│       ├── trainer.py               # Training loop
│       ├── evaluator.py             # Evaluation logic (metric computation)
│       ├── checkpoint/              # Checkpoint save/restore
│       └── callback/
│           ├── checkpoint_callback.py
│           ├── logging_callback.py
│           ├── eval_callback.py
│           └── early_stopping_callback.py
│
├── scripts/                         # Production entry points
│   ├── train/
│   │   ├── bert_finetuning.py
│   │   └── wav2vec_finetuning.py
│   ├── inference/
│   │   ├── bert_inference.py
│   │   └── wav2vec_inference.py
│   └── data_prep/
│       ├── convert_to_tfrecord.py
│       ├── convert_to_tfrecord_libri_light.py
│       ├── convert_to_tfrecord_librispeech.py
│       ├── create_librispeech_webdataset.py
│       ├── select_test_subset.py
│       └── process.py
│
├── configs/                         # Experiment configuration files
│
└── tests/                           # Integration / end-to-end tests
                                      # (unit tests stay next to source as *_test.py)
