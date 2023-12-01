# Ignore warnings for convergence during model training
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries
import datetime
import os
from functools import partial
import joblib
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
import numpy as np
import pandas as pd   # Import pandas with an alias for easy reference
import tensorflow as tf
print(tf.__version__)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from sklearn.mixture import BayesianGaussianMixture
from functools import partial
from sklearn.preprocessing import OneHotEncoder
import tensorflow_probability as tfp
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

import shutil
from functools import reduce
from tqdm.autonotebook import tqdm

# Install wget if not already installed
import wget

# from DataTransformers import *
# from GanActivation import *
# from GANEvaluation import *
# from Models import *
# from Utils import *
# from CTGan import *

class GenActivation(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, transformer_info, tau):
        """
        Custom Keras layer for generating activations in a neural network.

        Parameters:
        - input_dim: The input dimension of the layer.
        - output_dim: The output dimension of the layer.
        - transformer_info: Information about the transformation applied to specific segments of the output.
        - tau: Temperature parameter for Gumbel-Softmax during training.

        Initializes the layer with a dense transformation and Gumbel-Softmax activation.

        """
        super(GenActivation, self).__init__()
        self._output_dim = output_dim
        self._transformer_info = transformer_info
        self._tau = tau
        self._fc = tf.keras.layers.Dense(
            output_dim, input_dim=(input_dim,),
            kernel_initializer=partial(init_bounded, dim=input_dim),
            bias_initializer=partial(init_bounded, dim=input_dim))

    def call(self, inputs, training=False, **kwargs):
        """
        Forward pass of the layer.

        Parameters:
        - inputs: The input tensor.
        - training: Boolean indicating whether the model is in training mode.
        - **kwargs: Additional keyword arguments.

        Returns:
        - outputs: The output tensor from the dense layer.
        - data_t: Transformed data based on specified segments and Gumbel-Softmax activations.

        """
        outputs = self._fc(inputs, **kwargs)
        data_t = tf.zeros(tf.shape(outputs))
        for idx in self._transformer_info:
            if idx[2] == 0:
                act = tf.math.tanh(outputs[:, idx[0]:idx[1]])
            else:
                act = self._gumbel_softmax(outputs[:, idx[0]:idx[1]], self._tau, training=training)
            data_t = tf.concat([data_t[:, :idx[0]], act, data_t[:, idx[1]:]], 1)
        return outputs, data_t

    @tf.function(experimental_relax_shapes=True)
    def _gumbel_softmax(self, logits, tau=1.0, hard=False, training=True, dim=-1):
        """
        Gumbel-Softmax activation function.

        Parameters:
        - logits: Logit values before applying the softmax.
        - tau: Temperature parameter for controlling the level of randomness.
        - hard: Boolean indicating whether to use hard Gumbel-Softmax during training.
        - training: Boolean indicating whether the model is in training mode.
        - dim: Dimension along which softmax is applied.

        Returns:
        - y: Gumbel-Softmax activated output.

        """
        if not training:
            # When not training, use hard Gumbel-Softmax to remove randomness
            hard = True

        if hard:
            # Use a straight-through Gumbel-Softmax (hard)
            y = tf.nn.softmax(logits / tau, axis=dim)
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, axis=dim, keepdims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
            return y
        else:
            # Standard Gumbel-Softmax operation (soft, with randomness)
            gumbel_dist = tfp.distributions.Gumbel(loc=0, scale=1)
            gumbels = gumbel_dist.sample(tf.shape(logits))
            gumbels = (logits + gumbels) / tau
            y = tf.nn.softmax(gumbels, dim)
            return y


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        """
        Custom Keras layer implementing a residual connection with dense, batch normalization, and ReLU operations.

        Parameters:
        - input_dim: The input dimension of the layer.
        - output_dim: The output dimension of the layer.

        Initializes the layer with a dense transformation, batch normalization, and ReLU activation.

        """
        super(ResidualLayer, self).__init__()
        self._output_dim = output_dim
        self._fc = tf.keras.layers.Dense(
            self._output_dim,
            input_dim=(input_dim,),
            kernel_initializer=partial(init_bounded, dim=input_dim),
            bias_initializer=partial(init_bounded, dim=input_dim))
        self._bn = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)
        self._relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        """
        Forward pass of the residual layer.

        Parameters:
        - inputs: The input tensor.
        - **kwargs: Additional keyword arguments.

        Returns:
        - outputs: The output tensor after applying dense, batch normalization, and ReLU operations.
                   The final output is obtained by concatenating this result with the input tensor.

        """
        outputs = self._fc(inputs, **kwargs)
        outputs = self._bn(outputs, **kwargs)
        outputs = self._relu(outputs, **kwargs)
        return tf.concat([outputs, inputs], 1)

def init_bounded(shape, **kwargs):
    if 'dim' not in kwargs:
        raise AttributeError('dim not passed as input')
    if 'dtype' not in kwargs:
        raise AttributeError('dtype not passed as input')

    dim = kwargs['dim']
    d_type = kwargs['dtype']
    bound = 1 / math.sqrt(dim)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=d_type)