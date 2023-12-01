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
from Models import *
# from Utils import *
# from CTGan import *

class DataTransformer:

    @classmethod
    def from_dict(cls, in_dict):
        """
        Class method to create a new instance of DataTransformer from a dictionary.

        Parameters:
        - in_dict: Dictionary containing the attributes and values to initialize the new instance.

        Returns:
        - new_instance: An instance of DataTransformer initialized with values from the input dictionary.

        """
        new_instance = DataTransformer()
        new_instance.__dict__ = in_dict
        return new_instance

    def __init__(self, n_clusters=10, epsilon=0.005):
        """
        Initialize a DataTransformer instance with default or specified parameters.

        Parameters:
        - n_clusters: Number of clusters for data transformation (default is 10).
        - epsilon: Epsilon value for data transformation (default is 0.005).

        Initializes the instance with specified clustering parameters and additional attributes.

        """
        self._n_clusters = n_clusters
        self._epsilon = epsilon
        self._is_dataframe = None
        self._meta = None
        self._dtypes = None

        self.output_info = None
        self.output_dimensions = None
        self.output_tensor = None
        self.cond_tensor = None


    def generate_tensors(self):
        """
        Generate and set output and condition tensors based on the provided output information.

        Raises:
        - AttributeError: If output information is not available.

        Sets the `output_tensor` and `cond_tensor` attributes with TensorFlow constant tensors
        based on the specified output information.

        """
        if self.output_info is None:
            raise AttributeError("Output info still not available")

        output_info = []
        cond_info = []
        i = 0
        st_idx = 0
        st_c = 0
        for item in self.output_info:
            ed_idx = st_idx + item[0]
            if not item[2]:
                ed_c = st_c + item[0]
                cond_info.append(tf.constant(
                    [st_idx, ed_idx, st_c, ed_c, i], dtype=tf.int32))
                st_c = ed_c
                i += 1

            output_info.append(tf.constant(
                [st_idx, ed_idx, int(item[1] == 'softmax')], dtype=tf.int32))
            st_idx = ed_idx

        self.output_tensor = output_info
        self.cond_tensor = cond_info

    @ignore_warnings(category=ConvergenceWarning)
    def _fit_continuous(self, column, data):
        """
        Fit a Bayesian Gaussian Mixture (BGM) model to continuous data and extract relevant information.

        Parameters:
        - column: The name of the column being processed.
        - data: The continuous data for fitting the BGM model.

        Returns:
        - result_dict: Dictionary containing information about the fitted BGM model and its components.

        Fits a BGM model to the provided continuous data, identifies significant components, and
        returns a dictionary with relevant information for further processing.

        """
        vgm = BGM(
            n_components=self._n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        vgm.fit(data)
        components = vgm.weights_ > self._epsilon
        num_components = components.sum()

        return {
            'name': column,
            'model': vgm,
            'components': components,
            'output_info': [(1, 'tanh', 1), (num_components, 'softmax', 1)],
            'output_dimensions': 1 + num_components,
        }

    def _fit_discrete(self, column, data):
        """
        Fit a One-Hot Encoder (OHE) to discrete categorical data and extract relevant information.

        Parameters:
        - column: The name of the column being processed.
        - data: The discrete categorical data for fitting the OHE.

        Returns:
        - result_dict: Dictionary containing information about the fitted OHE encoder.

        Fits a One-Hot Encoder to the provided discrete categorical data, and
        returns a dictionary with relevant information for further processing.

        """
        ohe = OHE(sparse=False)
        ohe.fit(data)
        categories = len(ohe.categories_[0])

        return {
            'name': column,
            'encoder': ohe,
            'output_info': [(categories, 'softmax', 0)],
            'output_dimensions': categories
        }

    def fit(self, data, discrete_columns=tuple()):
        """
        Fit the DataTransformer to the provided data.

        Parameters:
        - data: The input data, either as a pandas DataFrame or as a 2D array-like object.
        - discrete_columns: A tuple containing the names of columns with discrete categorical data.

        Fits the DataTransformer to the input data, extracting information about each column
        based on whether it is continuous or discrete. The results are stored in instance attributes.

        """
        self.output_info = []
        self.output_dimensions = 0

        if not isinstance(data, pd.DataFrame):
            self._is_dataframe = False
            data = pd.DataFrame(data)
        else:
            self._is_dataframe = True

        self._dtypes = data.infer_objects().dtypes
        self._meta = []
        for column in data.columns:
            column_data = data[[column]].values
            if column in discrete_columns:
                meta = self._fit_discrete(column, column_data)
            else:
                meta = self._fit_continuous(column, column_data)

            self.output_info += meta['output_info']
            self.output_dimensions += meta['output_dimensions']
            self._meta.append(meta)

    def _transform_continuous(self, column_meta, data):
        """
        Transform continuous data using the fitted Bayesian Gaussian Mixture (BGM) model.

        Parameters:
        - column_meta: Dictionary containing information about the fitted BGM model for the column.
        - data: The continuous data to be transformed.

        Returns:
        - transformed_data: List containing transformed features and one-hot encoded probabilities.

        Transforms continuous data based on the provided BGM model information, selecting
        optimal components and applying appropriate transformations.

        """
        components = column_meta['components']
        model = column_meta['model']

        means = model.means_.reshape((1, self._n_clusters))
        stds = np.sqrt(model.covariances_).reshape((1, self._n_clusters))
        features = (data - means) / (4 * stds)

        probs = model.predict_proba(data)

        n_opts = components.sum()
        features = features[:, components]
        probs = probs[:, components]

        opt_sel = np.zeros(len(data), dtype='int')
        for i in range(len(data)):
            norm_probs = probs[i] + 1e-6
            norm_probs = norm_probs / norm_probs.sum()
            # opt_sel[i] = np.random.choice(np.arange(n_opts), p=norm_probs)
            opt_sel[i] = np.argmax(norm_probs)

        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99)

        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        return [features, probs_onehot]

    def _transform_discrete(self, column_meta, data):
        encoder = column_meta['encoder']
        return encoder.transform(data)

    def transform(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        values = []
        for meta in self._meta:
            column_data = data[[meta['name']]].values
            if 'model' in meta:
                values += self._transform_continuous(meta, column_data)
            else:
                values.append(self._transform_discrete(meta, column_data))

        return np.concatenate(values, axis=1).astype(float)

    def _inverse_transform_continuous(self, meta, data, sigma=None):
        model = meta['model']
        components = meta['components']

        mean = data[:, 0]
        variance = data[:, 1:]

        # if sigma is not None:
        #     mean = np.random.normal(mean, sigma)

        mean = np.clip(mean, -1, 1)
        v_t = np.ones((len(data), self._n_clusters)) * -100
        v_t[:, components] = variance
        variance = v_t
        means = model.means_.reshape([-1])
        stds = np.sqrt(model.covariances_).reshape([-1])
        p_argmax = np.argmax(variance, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        column = mean * 4 * std_t + mean_t

        return column

    def _inverse_transform_discrete(self, meta, data):
        encoder = meta['encoder']
        return encoder.inverse_transform(data)

    def inverse_transform(self, data, sigmas=None):
        """
        Inverse transform the generated data to the original space.

        Parameters:
        - data: The generated data to be inverse transformed.
        - sigmas: Optional parameter for specifying standard deviations for continuous variables.

        Returns:
        - output: The inverse transformed data.
        
        Inverse transforms the generated data to the original space based on the stored metadata.
        Handles both continuous and discrete variables.

        """
        start = 0
        output = []
        column_names = []
        for meta in self._meta:
            dimensions = meta['output_dimensions']
            columns_data = data[:, start:start + dimensions]

            if 'model' in meta:
                sigma = sigmas[start] if sigmas else None
                inverted = self._inverse_transform_continuous(
                    meta, columns_data, sigma)
            else:
                inverted = self._inverse_transform_discrete(meta, columns_data)

            output.append(inverted)
            column_names.append(meta['name'])
            start += dimensions

        output = np.column_stack(output)
        output = pd.DataFrame(output, columns=column_names)\
            .astype(self._dtypes)
        if not self._is_dataframe:
            output = output.values

        return output

class DataSampler:
    def __init__(self, data, output_info):
        """
        Initialize the DataSampler with data and output information.

        Parameters:
        - data: The input data.
        - output_info: Information about the model output.

        Initializes the DataSampler instance with data and output information.
        Constructs the model representation based on the output_info.

        """
        super(DataSampler, self).__init__()
        self._data = data
        self._model = []
        self._n = len(data)

        st_idx = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st_idx += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st_idx += item[0]
                    continue

                ed_idx = st_idx + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st_idx + j])[0])

                self._model.append(tmp)
                st_idx = ed_idx
            else:
                assert 0

        assert st_idx == data.shape[1]

    def sample(self, n_samples, col_idx, opt_idx):
        """
        Generate samples from the DataSampler.

        Parameters:
        - n_samples: The number of samples to generate.
        - col_idx: A list of column indices.
        - opt_idx: A list of indices representing the selected options.

        Returns:
        - sampled_data: The generated samples based on the specified column and option indices.

        Generates samples from the DataSampler based on the specified column and option indices.

        """
        if col_idx is None:
            idx = np.random.choice(np.arange(self._n), n_samples)
            return self._data[idx]

        idx = []
        for col, opt in zip(col_idx, opt_idx):
            idx.append(np.random.choice(self._model[col][opt]))

        return self._data[idx]

class ConditionalGenerator:
    @classmethod
    def from_dict(cls, in_dict):
        new_instance = ConditionalGenerator()
        new_instance.__dict__ = in_dict
        return new_instance

    def __init__(self, data=None, output_info=None, log_frequency=None):
        """
        Initialize the DataSampler with data, output information, and log frequency settings.

        Parameters:
        - data: The input data.
        - output_info: Information about the model output.
        - log_frequency: A boolean indicating whether to use log frequency.

        Initializes the DataSampler instance with data, output information, and log frequency settings.
        Constructs the model representation and calculates interval information based on the output_info.

        """
        if data is None or output_info is None or log_frequency is None:
            return

        self._model = []

        start = 0
        skip = False
        max_interval = 0
        counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                skip = True
                continue

            if item[1] == 'softmax':
                if skip:
                    skip = False
                    start += item[0]
                    continue

                end = start + item[0]
                max_interval = max(max_interval, end - start)
                counter += 1
                self._model.append(np.argmax(data[:, start:end], axis=-1))
                start = end

            else:
                assert 0

        assert start == data.shape[1]

        self._interval = []
        self._n_col = 0
        self.n_opt = 0
        skip = False
        start = 0
        self._p = np.zeros((counter, max_interval))
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                start += item[0]
                continue
            if item[1] == 'softmax':
                if skip:
                    start += item[0]
                    skip = False
                    continue
                end = start + item[0]
                tmp = np.sum(data[:, start:end], axis=0)
                if log_frequency:
                    tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self._p[self._n_col, :item[0]] = tmp
                self._interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self._n_col += 1
                start = end
            else:
                assert 0

        self._interval = np.asarray(self._interval)

    def _random_choice_prob_index(self, idx):
        prob = self._p[idx]
        rand = np.expand_dims(np.random.rand(prob.shape[0]), axis=1)
        return (prob.cumsum(axis=1) > rand).argmax(axis=1)

    def sample(self, batch_size):
        """
        Generate samples with conditional and mask information for the specified batch size.

        Parameters:
        - batch_size: The number of samples to generate.

        Returns:
        - cond: A numpy array representing one-hot encoded conditional information.
        - mask: A numpy array representing a binary mask indicating the selected columns.
        - col_idx: A numpy array containing randomly selected column indices.
        - opt_idx: A numpy array containing indices representing the selected options.

        Generates samples with one-hot encoded conditional information, a binary mask
        indicating the selected columns, and information about the selected options.

        """
        if self._n_col == 0:
            return None

        col_idx = np.random.choice(np.arange(self._n_col), batch_size)

        cond = np.zeros((batch_size, self.n_opt), dtype='float32')
        mask = np.zeros((batch_size, self._n_col), dtype='float32')

        mask[np.arange(batch_size), col_idx] = 1
        opt_idx = self._random_choice_prob_index(col_idx)
        opt = self._interval[col_idx, 0] + opt_idx
        cond[np.arange(batch_size), opt] = 1
        return cond, mask, col_idx, opt_idx

    def sample_zero(self, batch_size):
        """
        Generate zero-filled samples with random non-zero values for conditional generation.

        Parameters:
        - batch_size: The number of samples to generate.

        Returns:
        - vec: A numpy array containing zero-filled samples with randomly selected non-zero values.

        Generates samples with zero values and randomly selects non-zero values
        based on the information stored in the DataTransformer instance.

        """
        if self._n_col == 0:
            return None

        vec = np.zeros((batch_size, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self._n_col), batch_size)
        for i in range(batch_size):
            col = idx[i]
            pick = int(np.random.choice(self._model[col]))
            vec[i, pick + self._interval[col, 0]] = 1

        return vec