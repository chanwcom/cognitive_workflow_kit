package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_binary(
    name = "wav2vec_finetuning",
    srcs = [
        "wav2vec_finetuning.py",
    ],
    imports = [
      "..",
    ],
    deps = [
        "//data/format:speech_data_helper",
        "//loss/pytorch:seq_loss_util",
    ],
)

py_binary(
    name = "processor_test",
    srcs = [
        "processor_test.py",
    ],
    imports = [
      "..",
    ],
    deps = [
        "//data/format:speech_data_helper",
        "//loss/pytorch:seq_loss_util",
    ],
)

py_binary(
    name = "wav2vec_inference",
    srcs = [
        "wav2vec_inference.py",
    ],
    imports = [
      "..",
    ],
    deps = [
        "//data/format:speech_data_helper",
    ],
)

py_binary(
    name = "bert_finetuning",
    srcs = [
        "bert_finetuning.py",
    ],
    imports = [
      "..",
    ],
    deps = [
        "//data/format:speech_data_helper",
    ],
)

py_binary(
    name = "bert_inference",
    srcs = [
        "bert_inference.py",
    ],
    imports = [
      "..",
    ],
    deps = [
        "//data/format:speech_data_helper",
    ],
)
