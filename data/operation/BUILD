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
