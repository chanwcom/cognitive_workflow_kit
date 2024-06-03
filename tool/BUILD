package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_binary(
    name = "convert_to_tfrecord",
    srcs = [
        "convert_to_tfrecord.py",
    ],
    deps = [
        "//data/format:speech_data_pb2",
    ],
)

py_binary(
    name = "wav2vec_huggingface",
    srcs = [
        "wav2vec_huggingface.py",
    ],
    deps = [
        "//data/format:speech_data_helper",
    ],
)
