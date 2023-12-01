# Tabular GAN Anomaly Detection


This repository contains code for training a Tabular GAN (Generative Adversarial Network) for anomaly detection using TensorFlow. The Tabular GAN is used to generate synthetic samples that are similar to a given dataset, and it can be applied to detect anomalies by measuring the dissimilarity between the original and generated samples.


## Installation

To run this project, you need to set up a Python environment with the required dependencies. We recommend using a virtual environment to isolate project dependencies. Follow these steps to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/repo.git
cd your-repository
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
# Create a virtual environment (Python 3)
python3 -m venv venv # or just python, make changes accordingly

# Activate the virtual environment

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install project dependencies
# If you have both Python2.7 and Python3 installed, you might have to use pip3

pip install -r requirements.txt
```

This will install all the necessary libraries and versions specified in the `requirements.txt` file.

### 4. Run the Project

Now that the dependencies are installed, you can run the project using the following command:

```bash
python3 main.py
```

### Additional Notes:

- If you encounter any issues during installation, please refer to the [Troubleshooting](#troubleshooting) section.
- For advanced users or development, you may want to use [Docker](https://www.docker.com/) for containerization.


## Usage

1. **Get the Dataset**:
   The code first downloads a sample dataset from the URL specified in the code. This dataset will be used for training the GAN. You can change the dataset URL to your own dataset.

2. **Read the Dataset**:
   The downloaded dataset is then read using Pandas. Make sure that the dataset is in CSV format.

3. **Data Preprocessing**:
   The code performs data preprocessing by scaling the data using the MinMaxScaler. It assumes that the dataset contains both normal and anomaly samples, with 'class_label' indicating the class label.

4. **Train the GAN**:
   A Tabular GAN is built with the specified input dimension and noise dimension. The GAN is trained using the `train` method for a specified number of epochs and batch size.

5. **Generate Synthetic Samples for Anomaly Detection**:
   The code generates synthetic samples that are close to the original samples using the `generate_sample_close_to` method. It measures the Mean Squared Error (MSE) between the original and generated samples to determine the anomaly score.

6. **Evaluate Anomaly Detection**:
   The code tests the anomaly detection on both anomaly and non-anomaly samples and collects the anomaly scores. The anomaly detection performance is evaluated by plotting the distribution of anomaly scores for anomalies and non-anomalies.

## Sample Anomaly Detection

A sample anomaly detection is provided in the code for both anomaly and non-anomaly samples. It generates synthetic samples and calculates the MSE for 100 samples in each category, comparing them to the original samples.

## Results Visualization

The code also includes visualization of the density distribution of anomaly scores for anomalies and non-anomalies using Seaborn.

Please customize the code and data according to your specific use case. You can modify the number of epochs, batch size, and other hyperparameters to optimize the anomaly detection performance.


## Customization and Configuration

To adapt this code to your specific use case and dataset, you can make the following customizations and configurations:

1. **Dataset URL**: Update the `url` variable with the URL of your dataset. Make sure the dataset is in CSV format or modify the code to read data in a different format.

2. **Data Preprocessing**: If your dataset requires specific data preprocessing, you can modify the data preprocessing steps. Ensure that the data is scaled or transformed appropriately for your GAN.

3. **GAN Hyperparameters**: Adjust the GAN hyperparameters such as the number of epochs, batch size, and noise dimension in the `TabularGAN` class to achieve the best results for your dataset.

4. **Evaluation**: You can change the number of samples to test and other parameters in the evaluation section to fine-tune the anomaly detection.

5. **Visualizations**: Customize the visualizations, labels, and colors in the results visualization section to better understand the anomaly detection performance.

## Troubleshooting

If you encounter issues or have questions about using this code, please feel free to reach out to the project maintainers or seek assistance in relevant community forums or support channels.

## Acknowledgments

This code is based on the idea of using GANs for anomaly detection and leverages the capabilities of TensorFlow and other open-source libraries. We acknowledge the open-source community and contributors who have made this project possible.

## License

This code is provided under the [MIT License](LICENSE), which permits you to use, modify, and distribute the code for your own projects. Please review the license file for more details.

Feel free to adapt, improve, and extend this code to suit your anomaly detection needs. We hope this repository helps you in your anomaly detection projects. Good luck, and happy coding!
