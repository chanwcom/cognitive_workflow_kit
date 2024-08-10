"""A module for unit-testing layers used in Transformer."""

# pylint: disable=import-error

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
import tensorflow as tf
from google.protobuf import text_format
from packaging import version

# Custom imports
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers import attention
from speech.trainer.tf_based_end_to_end_trainer.tf_trainer.layers import normalization
from machine_learning.layers import conformer_block_layer_pb2
from machine_learning.layers import conformer_block_layer
from machine_learning.layers import dropout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

assert version.parse(tf.__version__) > version.parse("2.0.0"), (
    "At least tensorflow 2.0 is required.")


class ConformerBlockTest(tf.test.TestCase):
    """A class for unit-testing the FeedForwardModule class."""

    @classmethod
    def setUpClass(cls):
        """Creates the inputs to be used in this unit test."""
        cls._BATCH_SIZE = 3
        cls._MAX_SEQ_LEN = 40
        cls._MODEL_DIM = 32

        cls._seq_inputs = {}
        cls._seq_inputs["SEQ_LEN"] = tf.constant([33, 40, 21])
        mask = tf.expand_dims(tf.cast(
            tf.sequence_mask(cls._seq_inputs["SEQ_LEN"]), tf.dtypes.float32),
                              axis=2)
        cls._seq_inputs["SEQ_DATA"] = tf.random.uniform(
            shape=(cls._BATCH_SIZE, cls._MAX_SEQ_LEN, cls._MODEL_DIM)) * mask

    def test_output_size(self):
        """Tests the shape of the output of the layer."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            feed_forward_module_params: {
                activation_type: SWISH
                model_dim: 32
                feedforward_dim: 128
                dropout_params: {
                    seq_noise_shape: NONE
                    class_name: "BaselineDropout"
                    class_params: {
                        [type.googleapi.com/learning.BaselineDropoutParams] {
                            dropout_rate: 0.1
                        }
                    }
                }
            }

            mhsa_module_params: {
                model_dim: 32
                num_heads: 8
                relative_positional_embedding: False
                dropout_params: {
                    seq_noise_shape: NONE
                    class_name: "BaselineDropout"
                    class_params: {
                        [type.googleapi.com/learning.BaselineDropoutParams] {
                            dropout_rate: 0.1
                        }
                    }
                }
            }

            convolution_module_params: {
                conv_normalization_type: BATCH_NORM_WITH_MASK
                activation_type: SWISH
                model_dim: 32
                conv_kernel_size: 31
                dropout_params: {
                    seq_noise_shape: NONE
                    class_name: "BaselineDropout"
                    class_params: {
                        [type.googleapi.com/learning.BaselineDropoutParams] {
                            dropout_rate: 0.1
                        }
                    }
                }
            }
            """, conformer_block_layer_pb2.ConformerBlockParams())
        # yapf: enable
        layer = conformer_block_layer.ConformerBlock(params_proto)
        output = layer(self._seq_inputs)

        self.assertAllEqual(tf.shape(self._seq_inputs["SEQ_DATA"]),
                            tf.shape(output["SEQ_DATA"])),
        self.assertNotAllEqual(self._seq_inputs["SEQ_DATA"],
                               output["SEQ_DATA"])
        self.assertAllEqual(self._seq_inputs["SEQ_LEN"], output["SEQ_LEN"])

    def test_output_size_uniform_dist_dropout(self):
        # yapf: disable
        params_proto = text_format.Parse(
            """
            feed_forward_module_params: {
                activation_type: SWISH
                model_dim: 32
                feedforward_dim: 128
                dropout_params: {
                    seq_noise_shape: TIME
                    class_name: "UniformDistDropout"
                    class_params: {
                        [type.googleapi.com/learning.UniformDistDropoutParams] {
                            dropout_rate: 0.25
                        }
                    }
                }
            }

            mhsa_module_params: {
                model_dim: 32
                num_heads: 8
                relative_positional_embedding: False
                dropout_params: {
                    seq_noise_shape: TIME
                    class_name: "UniformDistDropout"
                    class_params: {
                        [type.googleapi.com/learning.UniformDistDropoutParams] {
                            dropout_rate: 0.25
                        }
                    }
                }
            }

            convolution_module_params: {
                conv_normalization_type: BATCH_NORM_WITH_MASK
                activation_type: SWISH
                model_dim: 32
                conv_kernel_size: 31
                dropout_params: {
                    seq_noise_shape: TIME
                    class_name: "UniformDistDropout"
                    class_params: {
                        [type.googleapi.com/learning.UniformDistDropoutParams] {
                            dropout_rate: 0.25
                        }
                    }
                }
            }
            """, conformer_block_layer_pb2.ConformerBlockParams())
        # yapf: enable
        layer = conformer_block_layer.ConformerBlock(params_proto)
        output = layer(self._seq_inputs)

        self.assertAllEqual(tf.shape(self._seq_inputs["SEQ_DATA"]),
                            tf.shape(output["SEQ_DATA"])),
        self.assertNotAllEqual(self._seq_inputs["SEQ_DATA"],
                               output["SEQ_DATA"])
        self.assertAllEqual(self._seq_inputs["SEQ_LEN"], output["SEQ_LEN"])


class FeedForwardModuleTest(tf.test.TestCase):
    """A class for unit-testing the FeedForwardModule class."""

    @classmethod
    def setUpClass(cls):
        """Creates the inputs to be used in this unit test."""
        cls._BATCH_SIZE = 3
        cls._MAX_SEQ_LEN = 40
        cls._MODEL_DIM = 32

        cls._seq_inputs = {}
        cls._seq_inputs["SEQ_LEN"] = tf.constant([33, 40, 21])
        mask = tf.expand_dims(tf.cast(
            tf.sequence_mask(cls._seq_inputs["SEQ_LEN"]), tf.dtypes.float32),
                              axis=2)
        cls._seq_inputs["SEQ_DATA"] = tf.random.uniform(
            shape=(cls._BATCH_SIZE, cls._MAX_SEQ_LEN, cls._MODEL_DIM)) * mask

    def test_default_values(self):
        """Tests whether default values are correctly set."""
        params_proto = conformer_block_layer_pb2.FeedForwardModuleParams()
        layer = conformer_block_layer.FeedForwardModule(params_proto)

        self.assertEqual(conformer_block_layer_pb2.SWISH,
                         layer._activation_type)
        self.assertEqual(512, layer._model_dim)
        self.assertEqual(2048, layer._feedforward_dim)
        self.assertEqual(0.1, layer._dropout_rate)

    def test_output_size(self):
        """Tests the shape of the output of the layer."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            activation_type: SWISH
            model_dim: 32
            feedforward_dim: 128
            dropout_params: {
                seq_noise_shape: NONE
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.2
                    }
                }
            }
            """, conformer_block_layer_pb2.FeedForwardModuleParams())
        # yapf: enable
        layer = conformer_block_layer.FeedForwardModule(params_proto)
        output = layer(self._seq_inputs)

        self.assertAllEqual(tf.shape(self._seq_inputs["SEQ_DATA"]),
                            tf.shape(output["SEQ_DATA"])),
        self.assertNotAllEqual(self._seq_inputs["SEQ_DATA"],
                               output["SEQ_DATA"])
        self.assertAllEqual(self._seq_inputs["SEQ_LEN"], output["SEQ_LEN"])

    def test_sub_layer_instances(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            activation_type: SWISH
            model_dim: 32
            feedforward_dim: 128
            dropout_params: {
                seq_noise_shape: NONE
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.2
                    }
                }
            }
            """, conformer_block_layer_pb2.FeedForwardModuleParams())
        # yapf: enable
        layer = conformer_block_layer.FeedForwardModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(6, len(layer._layers))

        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))
        self.assertTrue(isinstance(layer._layers[1], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[2], tf.keras.layers.Activation))
        self.assertTrue(isinstance(layer._layers[3], tf.keras.layers.Dropout))
        self.assertTrue(isinstance(layer._layers[4], tf.keras.layers.Dense))
        self.assertTrue(isinstance(layer._layers[5], tf.keras.layers.Dropout))

    def test_sub_layer_input_dropout_instances(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            activation_type: SWISH
            model_dim: 32
            feedforward_dim: 128
            dropout_params: {
                seq_noise_shape: BATCH
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.2
                    }
                }
            }
            input_dropout: True
            """, conformer_block_layer_pb2.FeedForwardModuleParams())
        # yapf: enable
        layer = conformer_block_layer.FeedForwardModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(6, len(layer._layers))

        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))
        self.assertTrue(isinstance(layer._layers[1], tf.keras.layers.Dropout))
        self.assertTrue(isinstance(layer._layers[2], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[3], tf.keras.layers.Activation))
        self.assertTrue(isinstance(layer._layers[4], tf.keras.layers.Dropout))
        self.assertTrue(isinstance(layer._layers[5], tf.keras.layers.Dense))

    def test_sub_layer_instances_uniform_dist_dropout(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            activation_type: SWISH
            model_dim: 32
            feedforward_dim: 128
            dropout_params: {
                seq_noise_shape: TIME
                class_name: "UniformDistDropout"
                class_params: {
                    [type.googleapi.com/learning.UniformDistDropoutParams] {
                        dropout_rate: 0.25
                    }
                }
            }
            """, conformer_block_layer_pb2.FeedForwardModuleParams())
        # yapf: enable
        layer = conformer_block_layer.FeedForwardModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(6, len(layer._layers))

        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))
        self.assertTrue(isinstance(layer._layers[1], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[2], tf.keras.layers.Activation))
        self.assertTrue(
            isinstance(layer._layers[3], dropout.UniformDistDropout))
        self.assertTrue(isinstance(layer._layers[4], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[5], dropout.UniformDistDropout))


class MHSAModuleTest(tf.test.TestCase):
    """A class for unit-testing the MHSAModule class."""

    @classmethod
    def setUpClass(cls):
        """Creates the inputs to be used in this unit test."""
        cls._BATCH_SIZE = 3
        cls._MAX_SEQ_LEN = 40
        cls._MODEL_DIM = 32

        cls._seq_inputs = {}
        cls._seq_inputs["SEQ_LEN"] = tf.constant([33, 40, 21])
        mask = tf.expand_dims(tf.cast(
            tf.sequence_mask(cls._seq_inputs["SEQ_LEN"]), tf.dtypes.float32),
                              axis=2)
        cls._seq_inputs["SEQ_DATA"] = tf.random.uniform(
            shape=(cls._BATCH_SIZE, cls._MAX_SEQ_LEN, cls._MODEL_DIM)) * mask

    def test_default_values(self):
        """Tests whether default values are correctly set."""
        params_proto = conformer_block_layer_pb2.MHSAModuleParams()
        layer = conformer_block_layer.MHSAModule(params_proto)
        self.assertEqual(512, layer._model_dim)
        self.assertEqual(8, layer._num_heads)
        self.assertEqual(True, layer._relative_positional_embedding)
        self.assertEqual(0.1, layer._dropout_rate)
        self.assertEqual(False, layer._causal)

    def test_output_size(self):
        """Tests the shape of the output of the layer."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            model_dim: 32
            num_heads: 8
            relative_positional_embedding: False
            dropout_params: {
                seq_noise_shape: NONE
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.1
                    }
                }
            }
            """, conformer_block_layer_pb2.MHSAModuleParams())
        # yapf: enable
        layer = conformer_block_layer.MHSAModule(params_proto)
        output = layer(self._seq_inputs)

        self.assertAllEqual(tf.shape(self._seq_inputs["SEQ_DATA"]),
                            tf.shape(output["SEQ_DATA"])),
        self.assertNotAllEqual(self._seq_inputs["SEQ_DATA"],
                               output["SEQ_DATA"])

    def test_sub_layer_instances(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            model_dim: 32
            num_heads: 8
            relative_positional_embedding: True
            dropout_params: {
                seq_noise_shape: NONE
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.1
                    }
                }
            }
            """, conformer_block_layer_pb2.MHSAModuleParams())
        # yapf: enable
        layer = conformer_block_layer.MHSAModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(3, len(layer._layers))

        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))

        self.assertTrue(
            isinstance(layer._layers[1], attention.RelativeAttention))
        self.assertTrue(isinstance(layer._layers[2], tf.keras.layers.Dropout))

    def test_sub_layer_input_dropout_instances(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            model_dim: 32
            num_heads: 8
            relative_positional_embedding: True
            dropout_params: {
                seq_noise_shape: BATCH
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.2
                    }
                }
            }
            input_dropout: True
            """, conformer_block_layer_pb2.MHSAModuleParams())
        # yapf: enable
        layer = conformer_block_layer.MHSAModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(3, len(layer._layers))

        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))
        self.assertTrue(isinstance(layer._layers[1], tf.keras.layers.Dropout))
        self.assertTrue(
            isinstance(layer._layers[2], attention.RelativeAttention))

    def test_sub_layer_instances(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            model_dim: 32
            num_heads: 8
            relative_positional_embedding: True
            dropout_params: {
                seq_noise_shape: TIME
                class_name: "UniformDistDropout"
                class_params: {
                    [type.googleapi.com/learning.UniformDistDropoutParams] {
                        dropout_rate: 0.25
                    }
                }
            }
            """, conformer_block_layer_pb2.MHSAModuleParams())
        # yapf: enable
        layer = conformer_block_layer.MHSAModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(3, len(layer._layers))

        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))

        self.assertTrue(
            isinstance(layer._layers[1], attention.RelativeAttention))
        self.assertTrue(
            isinstance(layer._layers[2], dropout.UniformDistDropout))


class ConvolutionModuleTest(tf.test.TestCase):
    """A class for unit-testing the FeedForwardModule class."""

    @classmethod
    def setUpClass(cls):
        """Creates the inputs to be used in this unit test."""
        cls._BATCH_SIZE = 3
        cls._MAX_SEQ_LEN = 40
        cls._MODEL_DIM = 32

        cls._seq_inputs = {}
        cls._seq_inputs["SEQ_LEN"] = tf.constant([33, 40, 21])
        mask = tf.expand_dims(tf.cast(
            tf.sequence_mask(cls._seq_inputs["SEQ_LEN"]), tf.dtypes.float32),
                              axis=2)
        cls._seq_inputs["SEQ_DATA"] = tf.random.uniform(
            shape=(cls._BATCH_SIZE, cls._MAX_SEQ_LEN, cls._MODEL_DIM)) * mask

    def test_default_values(self):
        """Tests whether default values are correctly set."""
        params_proto = conformer_block_layer_pb2.ConvolutionModuleParams()
        layer = conformer_block_layer.ConvolutionModule(params_proto)

        self.assertEqual(conformer_block_layer_pb2.BATCH_NORM_WITH_MASK,
                         layer._normalization_type)
        self.assertEqual(conformer_block_layer_pb2.SWISH,
                         layer._activation_type)
        self.assertEqual(512, layer._model_dim)
        self.assertEqual(31, layer._conv_kernel_size)
        self.assertEqual(0.1, layer._dropout_rate)
        self.assertEqual(False, layer._causal)

    def test_output_size(self):
        """Tests the shape of the output of the layer."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            conv_normalization_type: BATCH_NORM_WITH_MASK
            activation_type: SWISH
            model_dim: 32
            conv_kernel_size: 31
            dropout_params: {
                seq_noise_shape: BATCH
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.1
                    }
                }
            }
            """, conformer_block_layer_pb2.ConvolutionModuleParams())
        # yapf: enable
        layer = conformer_block_layer.ConvolutionModule(params_proto)
        output = layer(self._seq_inputs)

        self.assertAllEqual(tf.shape(self._seq_inputs["SEQ_DATA"]),
                            tf.shape(output["SEQ_DATA"])),
        self.assertNotAllEqual(self._seq_inputs["SEQ_DATA"],
                               output["SEQ_DATA"])

    def test_sub_layer_instances(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            conv_normalization_type: BATCH_NORM_WITH_MASK
            activation_type: SWISH
            model_dim: 32
            conv_kernel_size: 31
            dropout_params: {
                seq_noise_shape: BATCH
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.1
                    }
                }
            }
            """, conformer_block_layer_pb2.ConvolutionModuleParams())
        # yapf: enable
        layer = conformer_block_layer.ConvolutionModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(9, len(layer._layers))

        # yapf: disable
        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))
        self.assertTrue(
            isinstance(layer._layers[1], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[2], conformer_block_layer.GLU))
        self.assertTrue(
            isinstance(layer._layers[3], tf.keras.layers.Masking))
        self.assertTrue(
            isinstance(layer._layers[4], tf.keras.layers.DepthwiseConv1D))
        self.assertTrue(
            isinstance(layer._layers[5], normalization.BatchNormWithMask))
        self.assertTrue(
            isinstance(layer._layers[6], tf.keras.layers.Activation))
        self.assertTrue(
            isinstance(layer._layers[7], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[8], tf.keras.layers.Dropout))
        # yapf: enable

    def test_sub_layer_input_dropout_instances(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            conv_normalization_type: BATCH_NORM_WITH_MASK
            activation_type: SWISH
            model_dim: 32
            conv_kernel_size: 31
            dropout_params: {
                seq_noise_shape: BATCH
                class_name: "BaselineDropout"
                class_params: {
                    [type.googleapi.com/learning.BaselineDropoutParams] {
                        dropout_rate: 0.2
                    }
                }
            }
            input_dropout: True
            """, conformer_block_layer_pb2.ConvolutionModuleParams())
        # yapf: enable
        layer = conformer_block_layer.ConvolutionModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(11, len(layer._layers))

        # yapf: disable
        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))
        self.assertTrue(
            isinstance(layer._layers[1], tf.keras.layers.Dropout))
        self.assertTrue(
            isinstance(layer._layers[2], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[3], conformer_block_layer.GLU))
        self.assertTrue(
            isinstance(layer._layers[4], tf.keras.layers.Masking))
        self.assertTrue(
            isinstance(layer._layers[5], tf.keras.layers.Dropout))
        self.assertTrue(
            isinstance(layer._layers[6], tf.keras.layers.DepthwiseConv1D))
        self.assertTrue(
            isinstance(layer._layers[7], normalization.BatchNormWithMask))
        self.assertTrue(
            isinstance(layer._layers[8], tf.keras.layers.Activation))
        self.assertTrue(
            isinstance(layer._layers[9], tf.keras.layers.Dropout))
        self.assertTrue(
            isinstance(layer._layers[10], tf.keras.layers.Dense))
        # yapf: enable

    def test_sub_layer_instances_uniform_dist_dropout(self):
        """Tests the types of sub-layers."""
        # yapf: disable
        params_proto = text_format.Parse(
            """
            conv_normalization_type: BATCH_NORM_WITH_MASK
            activation_type: SWISH
            model_dim: 32
            conv_kernel_size: 31
            dropout_params: {
                seq_noise_shape: TIME
                class_name: "UniformDistDropout"
                class_params: {
                    [type.googleapi.com/learning.UniformDistDropoutParams] {
                        dropout_rate: 0.25
                    }
                }
            }

            """, conformer_block_layer_pb2.ConvolutionModuleParams())
        # yapf: enable
        layer = conformer_block_layer.ConvolutionModule(params_proto)

        # Checks the number of layers.
        self.assertEqual(9, len(layer._layers))

        # yapf: disable
        # Checks the type of each layer.
        self.assertTrue(
            isinstance(layer._layers[0], tf.keras.layers.LayerNormalization))
        self.assertTrue(
            isinstance(layer._layers[1], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[2], conformer_block_layer.GLU))
        self.assertTrue(
            isinstance(layer._layers[3], tf.keras.layers.Masking))
        self.assertTrue(
            isinstance(layer._layers[4], tf.keras.layers.DepthwiseConv1D))
        self.assertTrue(
            isinstance(layer._layers[5], normalization.BatchNormWithMask))
        self.assertTrue(
            isinstance(layer._layers[6], tf.keras.layers.Activation))
        self.assertTrue(
            isinstance(layer._layers[7], tf.keras.layers.Dense))
        self.assertTrue(
            isinstance(layer._layers[8], dropout.UniformDistDropout))
        # yapf: enable


if __name__ == "__main__":
    tf.test.main()
