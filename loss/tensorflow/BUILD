package(
    default_visibility = ["//visibility:public"]
)

py_library(
    name = "seq_loss",
    srcs = [
        "seq_loss.py",
    ],
    deps = [
        ":seq_loss_util",
    ],
)

py_test(
    name = "seq_loss_test",
    srcs = [
        "seq_loss_test.py",
    ],
    deps = [
        ":seq_loss",
    ],
)

py_library(
    name = "seq_loss_util",
    srcs = [
        "seq_loss_util.py",
    ],
)

py_test(
    name = "seq_loss_util_test",
    srcs = [
        "seq_loss_util_test.py",
    ],
    deps = [
        ":seq_loss_util",
    ],
)
