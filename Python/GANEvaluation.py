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

class GANEvaluation:
    def __init__(self):
        # Define a list of distinct colors to be used for plotting different datasets
        self.colors = ['blue', 'orange', 'red', 'green', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'yellow']
        pass

    def plot_kde(self, data=()):
        """
        Generate and display Kernel Density Estimate (KDE) plots for real and generated data.

        Parameters:
        - data: Tuple of DataFrames, where each DataFrame represents a dataset.

        Raises:
        - ValueError: If no data is provided.
        - AssertionError: If the number and names of columns in different datasets are inconsistent.
        """
        # Create a KDE plot for each column in the provided datasets and visualize real and generated data distributions
        plt.figure(figsize=(15, 10))
        if len(data) == 0:
            raise ValueError('No data provided')

        for i in range(len(data)-1):
            assert len(data[i].columns) == len(data[i+1].columns)
            for j in range(len(data[i].columns)):
                assert data[i].columns[j] == data[i+1].columns[j]

        num_columns = len(data[0].columns)

        for i in range(num_columns):
            plt.subplot((num_columns // 3) + 1, 3, i+1)
            for j, df in enumerate(data):
                sns.kdeplot(df[df.columns[i]], shade=True, color=self.colors[j],
                            label=f'{"Real Data" if j == 0 else "Generated Data"}')
                plt.legend(prop={'size': 5})
        plt.show(block=False)

    def plot_losses(self, losses):
        """
        Plot and display the losses of the generator and discriminator over epochs.

        Parameters:
        - losses: Dictionary containing "g_loss" (Generator Loss) and "d_loss" (Discriminator Loss) values.

        Displays a line plot of generator and discriminator losses over epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(losses["g_loss"], label='Generator Loss')
        plt.plot(losses["d_loss"], label='Discriminator Loss')
        plt.title('GAN Losses Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show(block=False)
