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

from DataTransformers import *
from GanActivation import *
from GANEvaluation import *
from Models import *
from Utils import *
from CTGan import *

class CTGANSynthesizer:
    def __init__(self,
                 file_path=None,
                 log_dir=None,
                 z_dim=128,
                 pac=10,
                 gen_dim=(256, 256),
                 crt_dim=(256, 256),
                 l2_scale=1e-6,
                 batch_size=500,
                 gp_lambda=10.0,
                 tau=0.2):
        # pylint: disable=too-many-arguments, too-many-locals
        if file_path is not None:
            self._load(file_path)
            return
        if log_dir is not None and os.path.exists(log_dir):
            raise IsADirectoryError("Log directory does not exist.")
        if batch_size % 2 != 0 or batch_size % pac != 0:
            raise ValueError(
                "batch_size needs to be an even value divisible by pac.")

        self._log_dir = log_dir
        self._z_dim = z_dim
        self._pac = pac
        self._pac_dim = None
        self._l2_scale = l2_scale
        self._batch_size = batch_size
        self._gp_lambda = gp_lambda
        self._tau = tau
        self._gen_dim = tuple(gen_dim)
        self._crt_dim = tuple(crt_dim)
        self._g_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-5, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self._c_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-5, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self._transformer = DataTransformer()
        self._data_sampler = None
        self._cond_generator = None
        self._generator = None
        self._critic = None
        self._losses = {"g_loss":[], "d_loss":[]}

    def train(self,
              train_data,
              discrete_columns=tuple(),
              epochs=300,
              log_frequency=True):
        """
        Train the CTGAN synthesizer on the given data.

        Parameters:
        - train_data: The training dataset.
        - discrete_columns: Tuple of column names with discrete data.
        - epochs: Number of training epochs.
        - log_frequency: Whether to log training progress.

        Trains the CTGAN synthesizer on the provided data for the specified number of epochs.

        """

        self._losses = {"g_loss":[], "d_loss":[]}

        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        self._transformer.generate_tensors()

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info)
        data_dim = self._transformer.output_dimensions
        self._cond_generator = ConditionalGenerator(
            train_data,
            self._transformer.output_info,
            log_frequency)
        self._generator = Generator(
            self._z_dim + self._cond_generator.n_opt,
            self._gen_dim,
            data_dim,
            self._transformer.output_tensor,
            self._tau)
        self._critic = Critic(
            data_dim + self._cond_generator.n_opt,
            self._crt_dim,
            self._pac)

        # Create TF metrics
        metrics = {
            'g_loss': tf.metrics.Mean(),
            'cond_loss': tf.metrics.Mean(),
            'c_loss': tf.metrics.Mean(),
            'gp': tf.metrics.Mean(),
        }
        if self._log_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = \
                self._log_dir + '/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Build model graphs
        self._generator.build((self._batch_size, self._generator._input_dim))
        self._critic.build((self._batch_size, self._critic._input_dim))

        # Train models
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for epoch in range(epochs):
            p_bar = ProgressBar(
                len(train_data), self._batch_size, epoch, epochs, metrics)
            for _ in range(steps_per_epoch):
                c_loss, g_p = self._train_c()
                metrics['c_loss'](c_loss)
                metrics['gp'](g_p)
                g_loss, cond_loss = self._train_g()
                metrics['g_loss'](g_loss)
                metrics['cond_loss'](cond_loss)
                p_bar.update(metrics)

            self._losses["g_loss"].append(g_loss)
            self._losses["d_loss"].append(c_loss)

            if self._log_dir is not None:
                with train_summary_writer.as_default():
                    for met in metrics:
                        tf.summary.scalar(
                            met, metrics[met].result(), step=epoch)
                        metrics[met].reset_states()
            p_bar.close()
            del p_bar

    @tf.function
    def train_c_step(self, fake_cat, real_cat):
        with tf.GradientTape() as tape:
            y_fake = self._critic(fake_cat, training=True)
            y_real = self._critic(real_cat, training=True)

            g_p = gradient_penalty(
                partial(self._critic, training=True), real_cat, fake_cat,
                self._pac, self._gp_lambda)
            loss = -(tf.reduce_mean(y_real) - tf.reduce_mean(y_fake))
            c_loss = loss + g_p

        grad = tape.gradient(c_loss, self._critic.trainable_variables)
        self._c_opt.apply_gradients(
            zip(grad, self._critic.trainable_variables))
        return loss, g_p

    def _train_c(self):
        fake_z = tf.random.normal([self._batch_size, self._z_dim])

        # Generate data_modules vector
        cond_vec = self._cond_generator.sample(self._batch_size)
        if cond_vec is None:
            _, _, col_idx, opt_idx = None, None, None, None
            real = self._data_sampler.sample(
                self._batch_size, col_idx, opt_idx)
        else:
            cond, _, col_idx, opt_idx = cond_vec
            cond = tf.convert_to_tensor(cond)
            fake_z = tf.concat([fake_z, cond], 1)

            perm = np.arange(self._batch_size)
            np.random.shuffle(perm)
            real = self._data_sampler.sample(
                self._batch_size, col_idx[perm], opt_idx[perm])
            cond_perm = tf.gather(cond, perm)

        fake, fake_act = self._generator(fake_z, training=True)
        real = tf.convert_to_tensor(real.astype('float32'))

        if cond_vec is not None:
            fake_cat = tf.concat([fake_act, cond], 1)
            real_cat = tf.concat([real, cond_perm], 1)
        else:
            fake_cat = fake
            real_cat = real

        return self.train_c_step(fake_cat, real_cat)

    @tf.function
    def train_g_step(self, fake_z):
        with tf.GradientTape() as tape:
            _, fake_act = self._generator(fake_z, training=True)
            y_fake = self._critic(fake_act, training=True)
            g_loss = -tf.reduce_mean(y_fake)

        weights = self._generator.trainable_variables
        grad = tape.gradient(g_loss, weights)
        grad = [grad[i] + self._l2_scale * weights[i]
                for i in range(len(grad))]
        self._g_opt.apply_gradients(
            zip(grad, self._generator.trainable_variables))
        return g_loss, tf.constant(0, dtype=tf.float32)

    @tf.function
    def train_g_cond_step(self, fake_z, cond, mask, cond_info):
        with tf.GradientTape() as tape:
            fake, fake_act = self._generator(fake_z, training=True)
            y_fake = self._critic(
                tf.concat([fake_act, cond], 1), training=True)
            cond_loss = conditional_loss(cond_info, fake, cond, mask)
            g_loss = -tf.reduce_mean(y_fake) + cond_loss

        weights = self._generator.trainable_variables
        grad = tape.gradient(g_loss, weights)
        grad = [grad[i] + self._l2_scale * weights[i]
                for i in range(len(grad))]
        self._g_opt.apply_gradients(
            zip(grad, self._generator.trainable_variables))
        return g_loss, cond_loss

    def _train_g(self):
        fake_z = tf.random.normal([self._batch_size, self._z_dim])
        cond_vec = self._cond_generator.sample(self._batch_size)

        if cond_vec is None:
            return self.train_g_step(fake_z)

        cond, mask, _, _ = cond_vec
        cond = tf.convert_to_tensor(cond, name="c1")
        mask = tf.convert_to_tensor(mask, name="m1")
        fake_z = tf.concat([fake_z, cond], 1, name="fake_z")
        return self.train_g_cond_step(
            fake_z, cond, mask, self._transformer.cond_tensor)

    def sample(self, n_samples):
        """
        Generate synthetic samples using the trained CTGAN synthesizer.

        Parameters:
        - n_samples: Number of synthetic samples to generate.

        Returns:
        - Synthetic samples generated by the CTGAN synthesizer.

        Generates synthetic samples using the trained CTGAN synthesizer. The number of generated samples
        is specified by the `n_samples` parameter.
        """
        if n_samples <= 0:
            raise ValueError("Invalid number of samples.")

        steps = n_samples // self._batch_size + 1
        data = []
        for _ in tf.range(steps):
            fake_z = tf.random.normal([self._batch_size, self._z_dim])
            cond_vec = self._cond_generator.sample_zero(self._batch_size)
            if cond_vec is not None:
                cond = tf.constant(cond_vec)
                fake_z = tf.concat([fake_z, cond], 1)

            fake = self._generator(fake_z)[1]
            data.append(fake.numpy())

        data = np.concatenate(data, 0)
        data = data[:n_samples]
        return self._transformer.inverse_transform(data, None)

    def single_sample(self, n_samples=1, noise=None):
        """
        Generate a single synthetic sample using the trained CTGAN synthesizer.

        Parameters:
        - n_samples: Number of synthetic samples to generate (default is 1).
        - noise: Noise vector for generating the synthetic sample (default is None).

        Returns:
        - A single synthetic sample generated by the CTGAN synthesizer.

        Generates a synthetic sample using the trained CTGAN synthesizer. The number of samples to
        generate is specified by the `n_samples` parameter, and a custom noise vector can be provided
        through the `noise` parameter.
        """
        if n_samples <= 0:
            raise ValueError("Invalid number of samples.")

        # Ensure noise is provided and has the right shape
        if noise is None:
            noise = tf.zeros([self._batch_size, self._z_dim])
        else:
            noise = tf.convert_to_tensor(noise)
            noise = tf.reshape(noise, [1, self._z_dim])

        # Generate the fake data
        fake = self._generator(noise, training=False)[1]
        data = fake.numpy()
        return self._transformer.inverse_transform(data, None)

    def generate_sample_close_to(self, x, num_iterations=2000, learning_rate=0.01, patience=10, reduce_factor=0.1, min_delta=1e-8, min_lr=0.0000001, verbose=False):
      """
        Generate a synthetic sample close to the given input sample.

        Parameters:
        - x: Input sample for which a close synthetic sample is generated.
        - num_iterations: Number of optimization iterations (default is 2000).
        - learning_rate: Initial learning rate for optimization (default is 0.01).
        - patience: Number of iterations with no improvement to wait before reducing learning rate (default is 10).
        - reduce_factor: Factor to reduce learning rate when patience is reached (default is 0.1).
        - min_delta: Minimum change in loss to be considered an improvement (default is 1e-8).
        - min_lr: Minimum learning rate allowed during optimization (default is 0.0000001).
        - verbose: If True, print information about learning rate reductions (default is False).

        Returns:
        - A numpy array representing the optimized noise vector for generating a synthetic sample close to the input.

        Generates a synthetic sample close to the given input sample by optimizing a noise vector. The optimization process
        is performed for a specified number of iterations, adjusting the learning rate and patience criteria dynamically.
        """

      # Transform the input sample x using the transformer
      transformed_x = self._transformer.transform(x)
      transformed_x = tf.constant(transformed_x, dtype=tf.float32)

      noise = tf.Variable(tf.random.normal((1, self._z_dim)), dtype=tf.float32)

      best_loss = float('inf')
      patience_counter = 0
      initial_learning_rate = learning_rate

      best_noise = noise

      optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

      alpha = 0.2

      for iteration in range(num_iterations):
          with tf.GradientTape() as tape:
              tape.watch(noise)
              generated_data, _ = self._generator(noise, training=False)  # Ensure deterministic output
              # Calculate the loss directly against the transformed data vector
              loss = tf.reduce_mean(tf.square(generated_data - transformed_x))


          gradients = tape.gradient(loss, noise)
          if gradients is None:
              raise RuntimeError('Gradients are None. This might be due to disconnected computational graph.')

          if (best_loss - loss.numpy()) < min_delta:
              patience_counter += 1
          else:
              patience_counter = 0
              best_loss = loss.numpy()
              best_noise = noise

          if patience_counter >= patience:
            noise = best_noise
            if learning_rate <= min_lr:
              break
            learning_rate *= reduce_factor
            optimizer.learning_rate = learning_rate
            if verbose: print(f"Reducing learning rate to {learning_rate:.7f} at iteration {iteration}")
            patience_counter = 0  # reset patience after reducing learning rate

          optimizer.apply_gradients([(gradients, noise)])

      return noise.read_value().numpy()  # Return the noise as a numpy array

    def get_losses(self):
      return self._losses


    def dump(self, file_path, overwrite=False):
        """
        Dump the CTGANSynthesizer instance to a file.

        Parameters:
        - file_path: The file path where the instance will be saved.
        - overwrite: If True, overwrite the file if it already exists (default is False).

        Saves the CTGANSynthesizer instance to a file using the joblib library.
        """
        if file_path is None or len(file_path) == 0:
            raise NameError("Invalid file_path.")
        dir_name = os.path.dirname(file_path)
        if len(dir_name) and not os.path.exists(os.path.dirname(file_path)):
            raise NotADirectoryError("The file directory does not exist.")
        if not overwrite and os.path.exists(file_path):
            raise FileExistsError(
                "File already exists. If you wish to replace it,"
                " use overwrite=True")

        # Create a copy of class dict as we are about to change the dictionary
        class_dict = {k: v for k, v in self.__dict__.items()
                      if type(v) in [int, float, tuple]}
        class_dict['_cond_generator'] = self._cond_generator.__dict__
        class_dict['_transformer'] = self._transformer.__dict__
        class_dict['_gen_weights'] = self._generator.get_weights()

        # Dump dictionary to file
        joblib.dump(class_dict, file_path)
        del class_dict

    def _load(self, file_path):
        if file_path is None or len(file_path) == 0:
            raise NameError("Invalid file_path.")
        if not os.path.exists(file_path):
            raise FileNotFoundError("The provided file_path does not exist.")

        # Load class attributes
        class_dict = joblib.load(file_path)
        if class_dict is None:
            raise AttributeError

        # Load class attributes
        for key, value in class_dict.items():
            if type(value) in [int, float, tuple]:
                setattr(self, key, value)

        # Load binary models/encoders to class dict
        self._transformer = DataTransformer.from_dict(
            class_dict['_transformer'])
        self._cond_generator = ConditionalGenerator.from_dict(
            class_dict['_cond_generator'])

        # Load Generator instance
        self._generator = Generator(
            self._z_dim + self._cond_generator.n_opt,
            self._gen_dim,
            self._transformer.output_dimensions,
            self._transformer.output_tensor,
            self._tau)
        self._generator.build((self._batch_size, self._generator._input_dim))
        self._generator.set_weights(class_dict['_gen_weights'])

def gradient_penalty(func, real, fake, pac=10, gp_lambda=10.0):
    """
    Calculate the gradient penalty for the specified function.

    Parameters:
    - func: The function for which the gradient penalty is calculated.
    - real: Real data.
    - fake: Generated data.
    - pac: The number of parallel augmentations.
    - gp_lambda: The weight of the gradient penalty.

    Returns:
    - gradient_penalty_value: The calculated gradient penalty value.

    Calculates the gradient penalty for the specified function using real and generated data,
    with the specified number of parallel augmentations and gradient penalty weight.

    """
    alpha = tf.random.uniform([real.shape[0] // pac, 1, 1], 0., 1.)
    alpha = tf.tile(alpha, tf.constant([1, pac, real.shape[1]], tf.int32))
    alpha = tf.reshape(alpha, [-1, real.shape[1]])

    interpolates = alpha * real + ((1 - alpha) * fake)
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        pred = func(interpolates)
    grad = tape.gradient(pred, [interpolates])[0]
    grad = tf.reshape(grad, tf.constant([-1, pac * real.shape[1]], tf.int32))

    slopes = tf.math.reduce_euclidean_norm(grad, axis=1)
    return tf.reduce_mean((slopes - 1.) ** 2) * gp_lambda