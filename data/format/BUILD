package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_proto_library(
    name = "speech_data_pb2",
    srcs = [
        "speech_data.proto"
    ],
    deps = [
         "@com_google_protobuf//:protobuf_python",
    ],
)

py(
     name = "speech_data_helper",
     srcs = [
         "speech_data_helper.py",
     ],
     deps = [
         ":speech_data_pb2",
     ],
)


py_test(
     name = "speech_data_helper_test",
     srcs = [
         "speech_data_helper_test.py",
     ],
     deps = [
         ":speech_data",
         ":speech_data_pb2",
     ],
)
