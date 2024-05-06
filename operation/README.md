# Operation framework

## Overview

A framework for performing numerical or behavioral operation.
[Footnote: A numerical operation means generating an output given an input. It
may be NumPy or Tensor operations]
[Footnote: A behavioral operation means something like performing model
training or creating directory structures, which does not necessarily return
output values.]

The operation specification is given in a protocol buffer messgae.






The basic functionality is giving input using "process"
# Creating a single "Operation".

TODO(chanw.com) Adds an example code and a separate page.



# Creating a dictionary of "Transforms"


# Creating a composite "Operations".

A useful scenario is building a network of transforms using composite
transforms.


  ## Handling the case when the rate of output is different from the input.



TODO(chanw.com Adds an example.

 * How to create


TODO(chanw.com) Explains the CompositeTransform


TODO(chanw.com)


