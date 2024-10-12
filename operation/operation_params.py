"""A module defining an AbstractOperationParams class.

This class is used as data structures containing parameters for iniitializing
Operations.

Example Usage:
"""

# pylint: disable=import-error, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import abc
from dataclasses import dataclass

@dataclass
class AbstractOpCreationParams(abc.ABC):
    """An abstract base class for creating all operation classes."""

@dataclass
class AbstractOpParams(abc.ABC):
    """An abstract base class for all operation parameter classes."""
