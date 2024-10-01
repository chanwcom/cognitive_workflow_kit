"""A module defining an AbstractOperation class.

A concrete class transforming input NumPy and/or Tensors are implemented by
deriving from this abstract class.
"""

# pylint: disable=import-error, no-member

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import abc
import platform
from enum import Enum

# Third-party imports
from packaging import version

# At least Python 3.4 is required because of the usage of Enum.
assert version.parse(platform.python_version()) > version.parse("3.4.0"), (
    "At least python verion 3.4 is required.")


class Message(Enum):
    """An enumeration type defining messages.

    Note that Message is the output of
    """
    # This message is generated by the source Operation when there is no more
    # input data remaining.
    END_OF_STREAM = 1

    # This message means that the current Operation needs more input data to
    # generate output. The OperationWrappers in the following module will
    # handle these cases.
    # //math_lib/transform/composite_transform.py
    NEED_MORE_INPUT_DATA = 2


class AbstractOperation(abc.ABC):
    """An abstract class for performing some "operations".

    Operations frequently include transformations on Tensorflow tensors or
    NumPy arrays.

    This is an interface from which users construct concrete classes for
    converting the input Tensors or NumPy arrays.

    Example Usage:

        import parameter_proto_pb2

        params_proto = parameter_proto_pb2.OperationParams()
        params_proto.parameter_a = 0.7
        params_proto.parameter_b = .12

        algorithm = create_transform(params_proto)
        output_tensor = algorithm.Process(inputs)
    """
    # TODO(chanw.com) Consider the following changes.
    # 1. The first parameter takes all of the following:
    #   params_proto (Either in unpacked or any proto)
    #   or params_dict. Both of them are supported.
    # 2. Remove transform_dict. It is only needed for CompositeOperation, but it
    # can be refactored in a way that operations in the transform_dict are
    # created internally.
    # yapf: disable
    @abc.abstractmethod
    def __init__(self, message=None,
            *args, **kwargs) -> None:
        # yapf: enable
        """Creates a Operation object.

        In most cases, it is recommended to initialize the class using
        "params_proto", that is a "Any" proto message. However, for
        simpler implementation, it is also possible to use "params_dict",
        which is a python dictionary.

        Args:
            params_proto: A proto-message for initialization.
                The type must be either of an "Any" or concrete message type.
            kwargs: The following keyword arguments are widely used.
                params_dict: Instead of using params_proto, it is possible to
                    specify the initialization information using Python dict.
                operations_dict: Used in the CompositeOperation.

        Returns:
            None
        """
        self._params_proto = None

        if params_proto is not None:
            # Calls the setter method of "params_proto".
            assert isinstance(params_proto, message.Message)
            self.params_proto = params_proto

    @property
    def params_proto(self):
        """Returns the params_proto proto-message used for initialization.

        Args:
            None.

        Returns:
            A proto-message containing the initialization information.
        """
        return self._params_proto

    @params_proto.setter
    def params_proto(self, params_proto):
        """Sets the params_proto proto-message for initialization.

        Args:
            params_proto: A proto-message containing the initialization info.

        Returns:
            None.
        """
        # Adds extra initialization information here.

        self._params_proto = params_proto

    @abc.abstractmethod
    def process(self, *args, **kwargs):
        """Processes the input array.

        Args:
            inputs: A tensor or NumPy array containing the input data.
                There may be "no" inputs as well.

        Returns:
            An object containing the processed data.
        """

        raise NotImplementedError
