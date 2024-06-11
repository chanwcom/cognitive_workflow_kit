"""A module defining the types of layers.

The following classes are implemented.
 * LayerType.
"""

# pylint: disable=no-member, import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanw.com@samsung.com)"

# Standard imports
import enum

# Third-party imports
import tensorflow as tf
from packaging import version

# TODO(chanw.com) Think about moving the location of the following util module.

assert version.parse(tf.__version__) >= version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class Type(enum.Enum):
    """An enumeration class defining the layer type."""
    # Layers for computation.
    DENSE = 1
    MULTI_HEAD_ATTENTION = 2
    CONV1D = 3
    DEPTHWISE_CONV1D = 4
    LSTM = 5
    SPEC_AUGMENT = 6
    # Note that sub-sampling may be implemented using various algorithms.
    SUB_SAMPLING = 7

    # Layers implementing various normalizations.
    LAYER_NORM = 10
    BATCH_NORM = 11
    BATCH_NORM_WITH_MASK = 12

    # Layers implementing various activations.
    ACTIVATION = 20

    # Layers implementing various regularizations.
    DROPOUT = 30
    UNIFORM_DIST_DROPOUT = 31

    # Layers for masking, padding or pooling.
    MASKING = 40
    PADDING = 41
    MAX_POOL = 42

    # Composite layers.
    CONFORMER_BLOCK = 100
