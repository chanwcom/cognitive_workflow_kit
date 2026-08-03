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
    name = "wav2vec_finetuning_paper",
    srcs = [
        "wav2vec_finetuning_paper.py",
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
    name = "wav2vec_finetuning_paper_1hr_exp",
    srcs = [
        "wav2vec_finetuning_paper_1hr_exp.py",
    ],
    imports = [
      "..",
    ],
    deps = [
        "//data/format:speech_data_helper",
        "//loss/pytorch:seq_loss_util_exp",
    ],
)

py_binary(
    name = "wav2vec_finetuning_paper_10hr",
    srcs = [
        "wav2vec_finetuning_paper_10hr.py",
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
    name = "wav2vec_finetuning_paper_100hr",
    srcs = [
        "wav2vec_finetuning_paper_100hr.py",
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
    name = "wav2vec_finetuning_paper_10min",
    srcs = [
        "wav2vec_finetuning_paper_10min.py",
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
        "//data/operation:text_codec",
    ],
)

py_binary(
    name = "tf_lite_test",
    srcs = [
        "tf_lite_test.py",
    ],
    imports = [
      "..",
    ],
    deps = [
        "//data/format:speech_data_helper",
        "//data/operation:text_codec",
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
