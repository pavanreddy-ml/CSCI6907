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

class ProgressBar(tqdm):
    @classmethod
    def _get_terminal_width(cls):
        """
        Get the terminal width, falling back to a default value if unavailable.

        Returns:
        - width: The terminal width.

        Gets the terminal width, falling back to a default value if unavailable.

        """
        width = shutil.get_terminal_size(fallback=(200, 24))[0]
        return width if width != 0 else 120

    def __init__(self, total_samples, batch_size, epoch, num_epochs, metrics):
        """
        Initialize the ProgressBar instance.

        Parameters:
        - total_samples: The total number of samples.
        - batch_size: The batch size.
        - epoch: The current epoch.
        - num_epochs: The total number of epochs.
        - metrics: The metrics to display in the progress bar.

        Initializes the ProgressBar instance with the specified parameters.

        """
        postfix = {m: f'{0:6.3f}' for m in metrics}
        postfix[1] = 1
        str_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  ' \
                     'ETA: {remaining}  Elapsed Time: {elapsed}  ' + \
                     reduce(lambda x, y: x + y,
                            ["%s:{postfix[%s]}  " % (m, m) for m in metrics],
                            "")
        super(ProgressBar, self).__init__(
            total=(total_samples // batch_size) * batch_size,
            ncols=int(ProgressBar._get_terminal_width() * .9),
            desc=tqdm.write(f'Epoch {epoch + 1}/{num_epochs}'),
            postfix=postfix,
            bar_format=str_format,
            unit='samples',
            miniters=10)
        self._batch_size = batch_size

    def update(self, metrics):
        """
        Update the progress bar with the latest metrics.

        Parameters:
        - metrics: The latest metrics.

        Updates the progress bar with the latest metrics.

        """
        for met in metrics:
            self.postfix[met] = f'{metrics[met].result():6.3f}'
        super(ProgressBar, self).update(self._batch_size)

def load_demo():
    """
    Load the demo dataset for testing.

    Returns:
    - demo_data: The loaded demo dataset.
    - discrete_columns: List of discrete columns in the dataset.

    Loads the demo dataset for testing, returning the dataset and a list of discrete columns.

    """
    demo_url = 'http://ctgan-data.s3.amazonaws.com/census.csv.gz'
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]
    return pd.read_csv(demo_url, compression='gzip'), discrete_columns

def get_test_variables():
    """
    Get test variables for model testing.

    Returns:
    - test_variables: Dictionary containing test variables.

    Retrieves a dictionary containing various test variables for model testing.

    """
    return {
        'decimal': 4,
        'input_dim': 10,
        'output_dim': 10,
        'pac': 10,
        'batch_size': 10,
        'gp_lambda': 10.0,
        'n_opt': 10,
        'n_col': 5,
        'layer_dims': [256, 256],
        'tau': 0.2
    }


def generate_data(batch_size, seed=0):
    """
    Generate synthetic data for testing.

    Parameters:
    - batch_size: The size of the synthetic dataset.
    - seed: The random seed for reproducibility.

    Returns:
    - synthetic_data: The generated synthetic dataset.
    - discrete_columns: List of discrete columns in the dataset.

    Generates synthetic data for testing, returning the dataset and a list of discrete columns.

    """
    np.random.seed(seed)
    data = np.concatenate((
        np.random.rand(batch_size, 1),
        np.random.randint(0, 5, size=(batch_size, 1))), axis=1)

    dataframe = pd.DataFrame(data, columns=['col1', 'col2'])
    discrete = ['col2']
    return dataframe, discrete