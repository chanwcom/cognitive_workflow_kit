package(
    default_visibility = ["//visibility:public"]
)

py_library(
    name = "seq_loss_util",
    srcs = [
        "seq_loss_util.py",
    ],
)

py_library(
    name = "seq_loss_util_exp",
    srcs = [
        "seq_loss_util_exp.py",
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
    legacy_create_init=True,
)
