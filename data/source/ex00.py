#!/usr/bin/python3

import soundfile

with soundfile.SoundFile("./testdata/011a0101.wav") as sound_file:
    data = sound_file.read(dtype="int16")

print (data)

