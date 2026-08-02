package(
    default_visibility = ["//visibility:public"]
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_library(
    name = "layer_type",
    srcs = [
        "layer_type.py",
    ],
)

py_proto_library(
    name = "dropout_params_pb2",
    srcs = [
        "dropout_params.proto",
    ],
    deps = [
        "@com_google_protobuf//:protobuf_python"
    ],
)

py_library(
    name = "masking_layer",
    srcs = [
        "masking_layer.py",
    ],
    deps = [
        "//math_lib/probability:empirical_distribution",
        "//math_lib/resize:resize_tensor",
    ],
)

py_test(
    name = "masking_layer_test",
    srcs = [
        "masking_layer_test.py",
    ],
    deps = [
        ":masking_layer",
        "//speech/trainer/util/tfdeterminism:determinism",
    ],
)

py_proto_library(
    name = "layer_params_pb2",
    srcs = [
        "layer_params.proto",
    ],
    deps = [
        "@com_google_protobuf//:protobuf_python",
        ":dropout_params_pb2",
    ],
)

py_library(
    name = "subsampling_layer",
    srcs = [
        "subsampling_layer.py",
    ],
    deps = [
        ":dropout",
        ":layer_params_pb2",
        ":layer_type",
        "//math_lib/operation:util",
        "//speech/trainer/ck_trainer/util:proto_util",
        "//speech/trainer/tf_based_end_to_end_trainer/tf_trainer/layers:layers",
    ],
)

py_test(
    name = "subsampling_layer_test",
    srcs = [
        "subsampling_layer_test.py",
    ],
    deps = [
        ":layer_params_pb2",
        ":subsampling_layer",
    ],
)

py_proto_library(
    name = "conformer_block_layer_pb2",
    srcs = [
        "conformer_block_layer.proto",
    ],
    deps = [
        ":dropout_params_pb2"
    ],
)

py_library(
    name = "conformer_block_layer",
    srcs = [
        "conformer_block_layer.py",
    ],
    deps = [
        ":conformer_block_layer_pb2",
        ":dropout",
        ":layer_type",
        "//speech/trainer/ck_trainer/util:proto_util",
        "//speech/trainer/tf_based_end_to_end_trainer/tf_trainer/layers:layers",
    ],
)

py_test(
    name = "conformer_block_layer_test",
    srcs = [
        "conformer_block_layer_test.py",
    ],
    deps = [
        ":conformer_block_layer",
        ":conformer_block_layer_pb2",
        "//speech/trainer/tf_based_end_to_end_trainer/tf_trainer/layers:layers",
    ],
)

py_library(
    name = "dropout",
    srcs = [
        "dropout.py",
    ],
    deps = [
        ":dropout_params_pb2",
        "//math_lib/operation:util",
        "//speech/trainer/ck_trainer/util:proto_util",
    ],
)

py_test(
    name = "dropout_test",
    srcs = [
        "dropout_test.py",
    ],
    deps = [
        ":dropout",
    ],
)

py_proto_library(
    name = "spec_augment_params_pb2",
    srcs = [
        "spec_augment_params.proto",
    ],
)

py_library(
    name = "spec_augment_layer",
    srcs = [
        "spec_augment_layer.py",
    ],
    deps = [
        ":spec_augment_params_pb2",
        "//math_lib/operation:operation",
        "//speech/trainer/ck_trainer/util:proto_util",
        "//speech/feature/spec_augmentation:spec_augmentation_tf",
    ],
)

py_test(
    name = "spec_augment_layer_test",
    srcs = [
        "spec_augment_layer_test.py",
    ],
    deps = [
        ":spec_augment_layer",
        "//speech/trainer/util/tfdeterminism:determinism",
    ],
)

#py_proto_library(
#    name = "neural_frontend_params_pb2",
#    srcs = [
#        "neural_frontend_params.proto",
#    ],
#)
