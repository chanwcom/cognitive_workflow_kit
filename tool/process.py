#!/usr/bin/python

#pylint: disable=

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import re

with open(sys.argv[1], encoding="utf-8") as file:
    for line in file:
        line = line.rstrip()

        # Removes the portion related to the goal and slots.
        transcript = re.sub(r"\[.*", "", line)
        words = transcript.split()

        # Removes words[0] ~ words[3] to retrieve the ground truth script.
        transcript = " ".join(words[4:]).upper()
        transcript = re.sub(r"\.", "", transcript)
        transcript = f"<s> {transcript} </s>"

        intent_slot = re.sub(r".*\[IN:", "", line)

        intent = intent_slot.split()[0].upper()

        match = re.match("\S+\s+(.*\S)\s*\]", intent_slot)
        assert match
        slot = match.group(1).upper()
        slot = re.sub(r"\.", "", slot)
        slot = f"<s> {slot} </s>"

        print(
            f"[{words[0]}] {transcript} __\"INTENT\"__:{intent}  __\"SLOT\"__:{slot}"
        )
