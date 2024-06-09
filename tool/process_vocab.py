#!/usr/bin/python3

print("{")
count = 0
line = True

with open("vocab", "rt") as file:
    while line:
        line = file.readline().rstrip()
        print(f"    \"{line}\": {count},")
        count += 1

print("}")
