# 🧠 Neural-Networks-and-Deep-Learning-Course
This repository contains the code for a university project on Neural Networks and Deep Learning. 
------------------------------------------------------------------------------------------------
# Assignment 1 – Neural Networks for Image Classification on CIFAR-10
The project explores the implementation, training, and evaluation of various Convolutional Neural Network (CNN) architectures for image classification using the CIFAR-10 dataset.

## Project Overview

The core task is to build and compare different neural network models to classify images from the CIFAR-10 dataset into 10 distinct categories. The notebooks cover data loading, preprocessing, model definition, training, and performance evaluation. Different architectures are explored, including variations in layers, dropout rates, and the use of parallel processing for training.

The general workflow in each notebook is as follows:

1. **Load Data**: The CIFAR-10 dataset is loaded.

2. **Preprocess Data**: Images are normalized, and labels are one-hot encoded. Data augmentation techniques are also applied to improve model generalization.

3. **Define Model**: Several CNN architectures are defined using the Keras Sequential API.

4. **Compile and Train**: The models are compiled with an optimizer (like Adam) and a loss function (categorical cross-entropy) and then trained on the dataset. Callbacks for early stopping and model checkpointing are used to optimize the training process.

5. **Evaluate**: The trained models are evaluated on the test set to measure their accuracy and loss.

6. **Visualize Results**: The training and validation accuracy and loss are plotted over epochs to visualize the learning process.

## Libraries and Frameworks

The project primarily relies on the following Python libraries and frameworks:

* **TensorFlow**: An end-to-end open-source platform for machine learning.

* **Keras**: A high-level neural networks API, running on top of TensorFlow, used for building and training the models.

* **Scikit-learn**: Used for splitting the data into training and validation sets.

* **NumPy**: A fundamental package for scientific computing with Python, used for numerical operations.

* **Matplotlib**: A plotting library used for creating static, animated, and interactive visualizations, particularly for plotting the training history.

* **Time & OS**: Standard Python libraries used for tracking training time and managing file paths.

## Code Files Summary

* `model_-1.ipynb`: This notebook represents a baseline or initial CNN model. It sets up the standard pipeline for data loading, preprocessing, model definition, training, and evaluation.

* `model_2.ipynb`: This file contains a more complex or refined version of the baseline model. It includes architectural changes like more convolutional layers, different dropout rates, or batch normalization to improve performance.

* `model_-1_parallel.ipynb`: This notebook explores distributed training. It adapts one of the models to be trained using TensorFlow's `MirroredStrategy`, which allows for training the model across multiple available GPUs in parallel to speed up the process.
-----------------------------------------------------------------------------------------------------------------------
# Assignment 2 – Classical Classifiers and MLP for Image Classification

This assignment shifts the focus from deep CNNs to traditional machine learning models and a basic Multi-Layer Perceptron (MLP), comparing their performance on the same CIFAR-10 dataset.

## Project Overview

The second assignment follows a similar logical workflow as the first one, including data loading, preprocessing, training, and evaluation. However, it introduces more intensive preprocessing steps like **Principal Component Analysis (PCA)** for dimensionality reduction and explores different classification algorithms.

The key differences and additions are:

* **Models Explored**: K-Nearest Neighbors (KNN), Nearest Centroid Classifier (NCC), Support Vector Machine (SVM), and a Multi-Layer Perceptron (MLP).

* **Preprocessing**: In addition to normalization, the data is flattened into 1D vectors, standardized using `StandardScaler`, and its dimensionality is reduced with `PCA`.

* **Hyperparameter Tuning**: This assignment heavily utilizes `GridSearchCV` for SVMs and manual grid searches for the MLP to find the best model parameters.

* **Visualization**: Introduces `seaborn` to create heatmaps for visualizing the results of hyperparameter tuning.

## Libraries and Frameworks

This assignment uses the same core libraries as the first one but with a heavier emphasis on **Scikit-learn** for its classifiers and preprocessing tools. New libraries introduced are:

* **Pandas**: Used for organizing the results from hyperparameter searches.

* **Seaborn**: For advanced data visualization, specifically heatmaps.

* **Joblib**: For saving and loading trained models (e.g., the optimized SVM).

## Code Files Summary

* `ToulkeridisNikolaosErgasia2_10718_KNN_NCC.py`: This script preprocesses the data with PCA and then implements and evaluates the K-Nearest Neighbors and Nearest Centroid classifiers.

* `ToulkeridisNikolaosErgasia2_10718.py`: This is the main script for the Support Vector Machine classifier. It performs an extensive hyperparameter search using `GridSearchCV` to find the best-performing SVM model.

* `ToulkeridisNikolaosErgasia2_10718_MLP.py`: Implements a Multi-Layer Perceptron. This script conducts a search over different numbers of neurons, learning rates, and batch sizes to optimize the simple neural network.

* `ToulkeridisNikolaosErgasia2_10718_Test.py`: A utility script to load a saved, trained model (like the SVM) and visualize its predictions on a sample of test images.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Assignment 3 – RBF Networks for Image Classification

This assignment introduces a different type of neural network, the Radial Basis Function (RBF) network. The goal is to build, train, and evaluate an RBF network for classifying images from the CIFAR-10 dataset, tackling both binary and multi-class classification problems.

## Project Overview

The third assignment follows a structured approach to implementing an RBF network. Unlike the previous assignments that used pre-built Keras layers, this one involves a more hands-on implementation, including the logic for the RBF layer itself.

The general workflow is as follows:
1.  **Data Selection & Preprocessing**: The CIFAR-10 dataset is loaded, and the user can select either two specific classes for a binary problem or all ten for a multi-class problem. The images are then converted to grayscale, flattened, and their dimensionality is reduced using PCA.
2.  **RBF Center Selection**: A key step in RBF networks is determining the centers for the radial basis functions. This is accomplished using the **K-Means clustering algorithm** on the training data.
3.  **Model Training**: With the centers defined, the RBF network is trained. This involves calculating the activations of the hidden layer and then training the weights of the output layer.
4.  **Evaluation and Visualization**: The trained model is evaluated on the test set using an accuracy score and a confusion matrix. The results are also visualized with examples of correct and incorrect classifications.

## Libraries and Frameworks

This assignment continues to use the core scientific computing and plotting libraries from the previous assignments. The most notable addition for the model's implementation is:

* **Scikit-learn**: Used extensively for preprocessing (`PCA`, `StandardScaler`) and, most importantly, for the `KMeans` clustering algorithm to find the RBF centers.

## Code Files Summary

* `Toulkeridis_Nikolaos_RBF.ipynb`: This notebook contains the complete implementation for the RBF network. It handles the entire pipeline from data loading and preprocessing to defining the RBF network, training it using K-Means for center selection, and finally evaluating and visualizing its performance on the CIFAR-10 test data.
--------------------------------------------------------------------------------------------------------------------------------------------------------------

This collection of notebooks provides a comprehensive exploration of building and optimizing Machine Learning Techniques for a standard image classification benchmark.
Important files (where to look):
- The Jupyter notebooks — open the .ipynb files to see the exact notebook-code and its results.
- Report PDFs — explains goals, experiments, and results.

Inputs:
- Dataset files referenced in the notebooks (see notebook cells to find dataset path and format).
- Notebook parameters (batch size, epochs, learning rate) defined in the code cells.

Outputs:
- Trained model weights (if saved by the notebooks).
- Plots: training/validation loss and accuracy, confusion matrix, example predictions.
- Numeric results: final accuracy, loss and short written conclusions in the PDF.

Usage:
- Open the .ipynb files in JupyterLab / Jupyter Notebook / Colab.
- Run cells top-to-bottom (install required packages first).
- To reproduce results, ensure the dataset path and any random seeds match what's set in the notebooks.

Notes / Caveats:
- Filenames and text include Greek; check the first cells of notebooks for exact package imports and environment setup.
