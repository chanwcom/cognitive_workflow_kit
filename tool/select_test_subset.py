#!/usr/bin/python3

import re
import numpy as np

np.random.seed(0)

file_name = "/home/chanwcom/speech_database/stop/test_0/music_test/manifest.tsv"


def get_num_lines(file_name: str):
    with open(file_name, "rb") as f:
        return sum(1 for _ in f)


num_lines = get_num_lines(file_name)
NUM_LINES_SELECTED = 300

index = np.sort(
    np.random.choice(np.arange(num_lines), NUM_LINES_SELECTED, replace=False))

j = 0
with open(file_name, "rt") as file:
    for i, line in enumerate(file):
        if i >= 0:  #== index[j]:
            line = line.rstrip()

            match = re.match(r"^(\S+)\s+(?:\S+\s+){3}([^\[]*)\[.*$", line)
            transcript = match.group(2).rstrip().upper()
            print(f"[{match.group(1)}] {transcript}")

            j += 1
            if j >= NUM_LINES_SELECTED:
                break
