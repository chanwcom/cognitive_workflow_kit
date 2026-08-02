package(default_visibility = ["//visibility:public"])

load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_python//python:proto.bzl", "py_proto_library")

py_proto_library(
    name = "speech_data_pb2",
    deps = [
        ":speech_data",
    ],
)

proto_library(
    name = "speech_data",
    srcs = [
        "speech_data.proto"
    ],
)

py_proto_library(
    name = "example_pb2",
    deps = [
         ":example",
    ],
)

proto_library(
    name = "example",
    srcs = [
        "example.proto"
    ],
)

py_library(
    name = "speech_data_helper",
    srcs = [
        "speech_data_helper.py",
    ],
    deps = [
        ":speech_data_pb2",
        "//operation:operation",
    ],
)

py_test(
     name = "speech_data_helper_test",
     srcs = [
         "speech_data_helper_test.py",
     ],
     deps = [
         ":speech_data_helper",
         ":speech_data_pb2",
     ],
)
