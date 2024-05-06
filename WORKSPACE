load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Protobuf fules for bazel.
#
# Refer to the following page about more information.
#
# https://thethoughtfulkoala.com/posts/2020/05/08/py-protobuf-bazel.html
#
# Comments:
#
# We have been using "rules_proto" that appears below to use
# "python_proto_library". However, as of May 2021, "python_proto_library" in
# "rules_proto" doesn't work well when there is a "__init__.py" in the target
# library.
http_archive(
    name = "com_google_protobuf",
    sha256 = "b10bf4e2d1a7586f54e64a5d9e7837e5188fc75ae69e36f215eb01def4f9721b",
    strip_prefix = "protobuf-3.15.3",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.15.3.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Protobuf rules for bazel.
#
# Refer to the following website for more information.
# https://github.com/bazelbuild/rules_proto.
# https://rules-proto-grpc.aliddell.com/en/latest/
#
# To be used with bazel4.0, the rules_proto commit version was upgraded
# from 97d8af4dc474595af3900dd85cb3a29ad28cc313 to
# cfdc2fa31879c0aebe31ce7702b1a9c8a4be02d2.
#
# Refer to the following page about the reason:
# https://github.com/bazelbuild/bazel/issues/12887
http_archive(
    name = "rules_proto",
    sha256 = "d8992e6eeec276d49f1d4e63cfa05bbed6d4a26cfe6ca63c972827a0d141ea3b",
    strip_prefix = "rules_proto-cfdc2fa31879c0aebe31ce7702b1a9c8a4be02d2",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_proto/archive/cfdc2fa31879c0aebe31ce7702b1a9c8a4be02d2.tar.gz",
        "https://github.com/bazelbuild/rules_proto/archive/cfdc2fa31879c0aebe31ce7702b1a9c8a4be02d2.tar.gz",
    ],
)
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()


# rules_grpc_proto
#
# As of Aug. 2020, this rule is much more stable than the default 
# py_proto_library. Please use "python_proto_library" in this rule.
#
# Refer to the following website for more information.
# https://github.com/rules-proto-grpc/rules_proto_grpc
http_archive(
    name = "rules_proto_grpc",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/3.1.1.tar.gz"],
    sha256 = "7954abbb6898830cd10ac9714fbcacf092299fda00ed2baf781172f545120419",
    strip_prefix = "rules_proto_grpc-3.1.1",
)

load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_toolchains", "rules_proto_grpc_repos")

rules_proto_grpc_toolchains()
rules_proto_grpc_repos()


# absl.
local_repository(
    # Name of the Abseil repository. 
    # This name is defined within Abseil's WORKSPACE file, in its `workspace()`
    # metadata.
    name = "com_google_absl",
    path = "./third_party/abseil/abseil-cpp/",
)

# re2.
local_repository(
    name = "com_google_re2",
    path = "./third_party/re2",
)

# googletest.
bind(
    name = "googletest",
    actual = "@com_google_googletest//:gtest",
)
http_archive(
    name = "com_google_googletest",
    sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
    strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
    urls = [
        "https://mirror.bazel.build/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
    ],
)

# glog.
bind(
    name = "glog",
    actual = "@com_github_glog_glog//:glog"
)
git_repository(
    name = "com_github_glog_glog",
    commit = "5c576f78c49b28d89b23fbb1fc80f54c879ec02e",
    remote = "https://github.com/google/glog.git",
)

# librosa.
bind(
    name = "librosa",
    actual = "@com_github_librosa//:librosa",
)
new_git_repository(
    name = "com_github_librosa",
    build_file = "//bazel:librosa.BUILD",
    commit = "f7c6482f4523d886cf80ebc529d30413e847d453",
    remote = "https://github.com/librosa/librosa.git",
)

# soundfile.
bind(
    name = "soundfile",
    actual = "@com_github_soundfile//:soundfile",
)
new_git_repository(
    name = "com_github_soundfile",
    build_file = "//bazel:soundfile.BUILD",
    commit = "8ebb523725e315f24c5592677353c43c4562be54",
    remote = "https://github.com/bastibe/python-soundfile.git"
)


# SWIG.
bind(
    name = "swig",
    actual = "@net_sourceforge_swig//:swig",
)
# Sources
#"https://mirror.bazel.build/ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
#"http://ufpr.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
#"http://pilotfiber.dl.sourceforge.net/project/swig/swig/swig-3.0.8/swig-3.0.8.tar.gz",
new_local_repository(
    name = "net_sourceforge_swig",
    path = "third_party/swig-3.0.8",
    build_file = "//bazel:swig.BUILD",
)

# PCRE.
http_archive(
    name = "pcre",
    build_file = "//bazel:pcre.BUILD",
    sha256 = "69acbc2fbdefb955d42a4c606dfde800c2885711d2979e356c0636efde9ec3b5",
    strip_prefix = "pcre-8.42",
    urls = [
        "https://mirror.bazel.build/ftp.exim.org/pub/pcre/pcre-8.42.tar.gz",
        "http://ftp.exim.org/pub/pcre/pcre-8.42.tar.gz",
    ],
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "6e16c8bc91b1310a44f3965e616383dbda48f83e8c1eaa2370a215057b00cabe",
    strip_prefix = "gflags-77592648e3f3be87d6c7123eb81cbad75f9aef5a",
    urls = [ 
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
        "https://github.com/gflags/gflags/archive/77592648e3f3be87d6c7123eb81cbad75f9aef5a.tar.gz",
    ],
)

# Used by //third_party/protobuf:protobuf_python
bind(
    name = "six",
    actual = "//third_party/py/six",
)

# pybind11 for bazel
#
# Refer to the following website for more information:
# https://github.com/pybind/pybind11_bazel
http_archive(
  name = "pybind11_bazel",
  sha256 = "75922da3a1bdb417d820398eb03d4e9bd067c4905a4246d35a44c01d62154d91",
  strip_prefix = "pybind11_bazel-203508e14aab7309892a1c5f7dd05debda22d9a5",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/203508e14aab7309892a1c5f7dd05debda22d9a5.zip"],
)

# pybind 11
# 
# The above pybind11 bazel still requires this library.
# Refer to the following website for more information:
# https://github.com/pybind/pybind11
http_archive(
  name = "pybind11",
  sha256 = "ead170c1a6c67a9d5554be19f41548e6f93f966c00e8d7aace46bd83000647f0",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.5",
  urls = ["https://github.com/pybind/pybind11/archive/v2.5.tar.gz"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
