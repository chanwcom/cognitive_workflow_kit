package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_library(
    name = "text_codec",
    srcs = [
        "text_codec.py",
        "text_codec_params.py"
    ],
    deps = [
        "//operation:operation",
    ],
)

py_test(
    name = "text_codec_test",
    srcs = [
        "text_codec_test.py",
    ],
    deps = [
        ":text_codec",
    ],
    data = [
        "testdata/model_unigram_256.model",
    ],
    legacy_create_init = False,
)

py_library(
    name = "tf2_dataset_op",
    srcs = [
        "tf2_dataset_op.py",
        "dataset_op_params.py"
    ],
    deps = [
        "//operation:operation",
    ],
)

py_test(
    name = "tf2_dataset_op_test",
    srcs = [
        "tf2_dataset_op_test.py",
    ],
    deps = [
        ":tf2_dataset_op",
    ],
    legacy_create_init = False,
)
