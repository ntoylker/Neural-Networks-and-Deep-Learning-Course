# 🧠 Neural-Networks-and-Deep-Learning-Course
This repository contains the code for a university project on Neural Networks and Deep Learning. 
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

This collection of notebooks provides a comprehensive exploration of building and optimizing CNNs for a standard image classification benchmark.
Important files (where to look):
- The Jupyter notebooks in folder: "1o - Κατασκευή Νευρωνικού Δικτύου CNN-MLP" — open the .ipynb files to see the exact notebook-code and its results.
- Report PDF: ΝΝ_ΕΡΓΑΣΙΑ_Νικος_Τουλκεριδης_10718.pdf — explains goals, experiments, and results.

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

