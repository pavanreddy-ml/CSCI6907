# Tabular GAN Anomaly Detection


This repository contains code for training a Tabular GAN (Generative Adversarial Network) for anomaly detection using TensorFlow. The Tabular GAN is used to generate synthetic samples that are similar to a given dataset, and it can be applied to detect anomalies by measuring the dissimilarity between the original and generated samples.


## Installation

To run this project, you need to set up a Python environment with the required dependencies. We recommend using a virtual environment to isolate project dependencies. Follow these steps to get started:

### 1. Clone the Repository

```bash
git clone https://github.com/repo.git # ToDo
cd your-repository # will update this later, left for blind review
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

### Additional Notes:

- If you encounter any issues during installation, please refer to the [Troubleshooting](#troubleshooting) section.
- For advanced users or development, you may want to use [Docker](https://www.docker.com/) for containerization.

<!-- ### 4. Run the Project

Now that the dependencies are installed, you can run the project using the following command:

```bash
python3 main.py
``` -->
## Code Structure

The repository provides two ways to execute the code:

1. **Python standalone file**
2. **Jupyter Notebook**

### Jupyter Notebook
There are two folders, names of which are self explanotary. If you are running using a Jupyter Notebook, just start executing cells.

### Using Python command through CLI


The python code is organized into several files and functions:

1. **DataTransformers.py**: Contains data transformation functions.
2. **GanActivation.py**: Defines custom activation functions for the GAN model.
3. **GANEvaluation.py**: Provides functions for evaluating the AnoGAN model.
4. **Models.py**: Defines the architecture of the GAN model.
5. **Utils.py**: Contains utility functions used throughout the code.
6. **CTGan.py**: Implements the Conditional Transformation GAN (CT-GAN) model.
   
The **main.py** script is the provided code file, which imports and utilizes functions from the mentioned files to implement the AnoGAN model.

Certainly! Let's go into more detail on the "Getting Started" section, breaking down each step and providing additional explanations:

## Getting Started

### 1. **Data Download:**

   The code automatically downloads the dataset from [this link](https://raw.githubusercontent.com/google/madi/master/src/madi/datasets/data/anomaly_detection_sample_1577622599.csv) if the file does not exist locally. This dataset serves as the input for the AnoGAN model. The URL is specified in the `url` variable, and the file is named "data.csv." If you already have the dataset or want to use a different one, you can replace the URL or provide your own dataset with the same structure.

   ```python
   url = "https://raw.githubusercontent.com/google/madi/master/src/madi/datasets/data/anomaly_detection_sample_1577622599.csv"
   if os.path.exists("data.csv"):
       print(f"The file exists. No need to download!")
   else:
       print(f"The file does not exist.")
       wget.download(url, 'data.csv')
   ```

### 2. **Data Preprocessing:**

   The code reads the dataset using Pandas and performs preprocessing steps, including removing unnecessary columns (`'dow'`, `'hod'`, `'class_label'`, `'Unnamed: 0'`). Additionally, it applies Min-Max scaling to normalize the features to the range of -1 to 1.

   ```python
   df = pd.read_csv("data.csv")
   df.drop(['dow', 'hod'], axis=1, inplace=True)

   ana_df = df[df['class_label'] == 0]
   ana_df.drop(['class_label', 'Unnamed: 0'], axis=1, inplace=True)
   df_single_class = df[df['class_label'] == 1]
   df_single_class.drop(['class_label', 'Unnamed: 0'], axis=1, inplace=True)

   scaler = MinMaxScaler(feature_range=(-1, 1))
   scaled_df = pd.DataFrame(scaler.fit_transform(df_single_class), columns=df_single_class.columns)

   ana_df = pd.DataFrame(scaler.transform(ana_df), columns=ana_df.columns)
   ```

### 3. **Baseline Model:**

   The code implements a baseline One-Class SVM model for comparison with the AnoGAN model. It uses the scikit-learn library to fit the One-Class SVM on the scaled dataset and evaluates its accuracy.

   ```python
   print("Starting the execution of baseline model")
   ocsvm = OneClassSVM(gamma='auto', nu=0.1)
   ocsvm.fit(scaled_df)
   predictions = ocsvm.predict(pd.concat([ana_df.iloc[:500], scaled_df[:500]]))
   predictions[predictions == -1] = 0
   baseline_model_accuracy = sklearn.metrics.accuracy_score(predictions, [0]*500 + [1]*500)
   print(f"Accuracy for the baseline model OneClass SVM is: {baseline_model_accuracy}")
   ```

### 4. **AnoGAN Model Training:**

   The code initializes the AnoGAN model (`CTGANSynthesizer()`) and trains it on the preprocessed data for a specified number of epochs (in this case, 100 epochs).

   ```python
   print("Starting the execution of CT-GAN")
   model = CTGANSynthesizer()
   model.train(scaled_df, epochs=100)
   ```

### 5. **Parameters:**

These parameters can be changed according to the data. If you are testing the working of code, reduce the NI and N to values 100 and 10 respectively. This will ensure that the code can be tested in less time.
```
NI (Number of Iterations): 1000
LR (Learning Rate): 0.1
N (Number of Samples): 50
base: Varies from 10000 to 100000000 in steps of 1000000 for threshold optimization.
```

### 6. **Evaluation and Visualization:**

   The code then uses the `GANEvaluation` class to plot losses and kernel density estimates for the synthetic data generated by the AnoGAN model.

   ```python
   GANEvaluation().plot_losses(model.get_losses())
   synthetic_data = model.sample(10000)
   GANEvaluation().plot_kde(data=[scaled_df, synthetic_data])
   ```

### 7. **Anomaly Detection:**

   The code demonstrates the AnoGAN model's ability to generate synthetic data close to the original samples and calculates the mean squared error (MSE) as an anomaly score.

   `This example generates just one Anomaly and test, there is a part of code that generates multiple anomalies and then test it which is not included in this READme`

   ```python
   ana_sample = ana_df.head(1)   # Get 1 Anomaly sample for test
   nor_sample = scaled_df.head(1)     # Get 1 Normal sample for test

   # Get the noise vector that generates a sample closes to the original sample. Compare the samples and get anomaly score (MSE)
   print("Get the noise vector that generates an anomaly sample close to the original sample")
   x = model.generate_sample_close_to(ana_sample, learning_rate=2)
   generated_sample = model.single_sample(1, noise=x)
   mse = tf.reduce_mean(tf.abs(generated_sample - ana_sample))
   print(mse)

   # Get the noise vector that generates a sample closes to the original sample. Compare the samples and get anomaly score (MSE)
   print("Get the noise vector that generates a normal sample close to the original sample")
   x = model.generate_sample_close_to(nor_sample, learning_rate=2)
   generated_sample = model.single_sample(1, noise=x)
   mse = tf.reduce_mean(tf.abs(generated_sample - nor_sample))
   print(mse)
   ```

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
