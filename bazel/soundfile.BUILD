package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # ISC License

exports_files(["LICENCE"])

genrule(
    name = "build_wheels",
    srcs = [
        "setup.py",
        "soundfile.py",
        "soundfile_build.py",
        "README.rst",
    ],
    outs = ["_soundfile.py"],
    # TODO(chanwcom) Couldn't we go to the location first rather than copying
    # files to the working directory?
    cmd = """cp $(SRCS) . && python setup.py bdist_wheel &&
             cp ./build/lib/_soundfile.py $@""",
)


py_library(
    name = "soundfile",
    srcs = [
        "_soundfile.py",
        "soundfile.py",
    ],
)


