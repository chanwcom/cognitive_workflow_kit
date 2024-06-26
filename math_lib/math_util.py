#!/usr/bin/python3

# This should work both for Tensorflow, PyTorch, and NumPy arrays.

# The following needs to be refined.
# There are several glitches.
def print_line(inputs, precision, minf_value=None, minf_string="MINF",
               inf_value=None, inf_string="INF"):
    # It should be a rank-1 tensor.
    print ("[", end="")
    for i in range(len(inputs) - 1):
        if minf_value is not None and inputs[i] <= minf_value:
            print (f"{minf_string}, ", end="")
        elif inf_value is not None and inputs[i] > inf_value:
            print (f"{inf_string}, ", end="")
        else:
            print (f"{inputs[i]:10.{precision}f}, ", end="")
    if minf_value is not None and inputs[-1] <= minf_value:
        print (f"{minf_string}]", end="")
    elif inf_value is not None and inputs[-1] > inf_value:
        print (f"{inf_string}]", end="")
    else:
        print (f"{inputs[-1]:10.{precision}f}]", end="")

def print_array(inputs, precision, minf_value=None, minf_string="MINF",
                inf_value=None, inf_string="INF"):
    if len(inputs.shape) >= 2:
        print ("[", end="")

        for i, data in enumerate(inputs):
            print_array(data, precision, minf_value, minf_string, inf_value, inf_string)
            if i != len(inputs) - 1:
                print (",")
        print("]")
    else:
        print_line(inputs, precision, minf_value, minf_string, inf_value, inf_string)
