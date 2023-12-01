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
from GanActivation import *
# from GANEvaluation import *
# from Models import *
# from Utils import *
# from CTGan import *

class BGM(BayesianGaussianMixture):
    def __eq__(self, other):
        """
        Custom equality comparison for BayesianGaussianMixture instances.

        Parameters:
        - other: Another instance of BayesianGaussianMixture for comparison.

        Returns:
        - bool: True if the instances are equal based on their dictionaries, False otherwise.

        """
        try:
            np.testing.assert_equal(self.__dict__, other.__dict__)
            return True
        except AssertionError:
            return False

class Critic(tf.keras.Model):
    def __init__(self, input_dim, dis_dims, pac):
        """
        Custom Keras model representing the critic in a GAN architecture.

        Parameters:
        - input_dim: The input dimension of the model.
        - dis_dims: List of integers representing dimensions of discriminator layers.
        - pac: The number of parallel paths in the model.

        Initializes the model with dense layers, LeakyReLU activations, and dropout.

        """
        super(Critic, self).__init__()
        self._pac = pac
        self._input_dim = input_dim

        self._model = [self._reshape_func]
        dim = input_dim * self._pac
        for layer_dim in list(dis_dims):
            self._model += [
                tf.keras.layers.Dense(
                    layer_dim, input_dim=(dim,),
                    kernel_initializer=partial(init_bounded, dim=dim),
                    bias_initializer=partial(init_bounded, dim=dim)),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(0.5)]
            dim = layer_dim

        layer_dim = 1
        self._model += [tf.keras.layers.Dense(
            layer_dim, input_dim=(dim,),
            kernel_initializer=partial(init_bounded, dim=dim),
            bias_initializer=partial(init_bounded, dim=dim))]

    def _reshape_func(self, inputs, **kwargs):
        """
        Reshape function to transform input tensor dimensions.

        Parameters:
        - inputs: The input tensor.
        - **kwargs: Additional keyword arguments.

        Returns:
        - reshaped_inputs: The reshaped input tensor.

        """
        dims = inputs.get_shape().as_list()
        return tf.reshape(inputs, [-1, dims[1] * self._pac])

    def call(self, inputs, **kwargs):
        """
        Forward pass of the critic model.

        Parameters:
        - inputs: The input tensor.
        - **kwargs: Additional keyword arguments.

        Returns:
        - outputs: The output tensor from the critic model.

        """
        outputs = inputs
        for layer in self._model:
            outputs = layer(outputs, **kwargs)
        return outputs

class Generator(tf.keras.Model):
    def __init__(self, input_dim, gen_dims, data_dim, transformer_info, tau):
        """
        Custom Keras model representing the generator in a GAN architecture.

        Parameters:
        - input_dim: The input dimension of the model.
        - gen_dims: List of integers representing dimensions of generator layers.
        - data_dim: The dimension of the generated data.
        - transformer_info: Information about the transformation applied to the generated data.
        - tau: Temperature parameter for Gumbel-Softmax during training.

        Initializes the model with ResidualLayer and GenActivation layers.

        """
        super(Generator, self).__init__()

        self._input_dim = input_dim
        self._model = list()
        dim = input_dim
        for layer_dim in list(gen_dims):
            self._model += [ResidualLayer(dim, layer_dim)]
            dim += layer_dim

        self._model += [GenActivation(dim, data_dim, transformer_info, tau)]

    def call(self, inputs, **kwargs):
        """
        Forward pass of the generator model.

        Parameters:
        - inputs: The input tensor.
        - **kwargs: Additional keyword arguments.

        Returns:
        - outputs: The output tensor from the generator model.

        """
        outputs = inputs
        for layer in self._model:
            outputs = layer(outputs, **kwargs)
        return outputs

class OHE(OneHotEncoder):
    def __eq__(self, other):
        """
        Custom equality comparison for OneHotEncoder instances.

        Parameters:
        - other: Another instance of OneHotEncoder for comparison.

        Returns:
        - bool: True if the instances are equal based on their dictionaries, False otherwise.

        """
        try:
            np.testing.assert_equal(self.__dict__, other.__dict__)
            return True
        except AssertionError:
            return False
