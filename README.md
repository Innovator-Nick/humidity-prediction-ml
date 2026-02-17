# Humidity Prediction using Machine Learning (MLP Neural Network)

A time series machine learning project that predicts **relative humidity one month ahead** using meteorological data and a Multi-Layer Perceptron (MLP) neural network built with TensorFlow and Scikit-Learn.

This project demonstrates end-to-end machine learning workflow including preprocessing, feature engineering, model building, and evaluation.

---

# Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features and Target](#features-and-target)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Overfitting Prevention Techniques](#overfitting-prevention-techniques)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [License](#license)

---

# Project Overview

The objective of this project is to predict **relative humidity for the next month** using historical meteorological data.

This is a **time series regression problem** solved using a **Multi-Layer Perceptron Neural Network**.

The project covers:

- Data preprocessing
- Feature selection
- Neural network implementation
- Model evaluation
- Performance analysis

This project was developed as part of a university machine learning coursework.

---

# Dataset

Total instances:

6647

Train-Validation-Test Split:

- Training: 64%
- Validation: 16%
- Testing: 20%

The dataset contains meteorological measurements used to predict humidity.

---

# Features and Target

## Input Features

- groundfrost_2
- psl_2
- pv_2
- rainfall_2
- sfcWind_2
- snowLying_2
- sun_2
- tas_2

## Target Variable

- hurs_2

Relative humidity of the next month.

---

# Project Structure

humidity-prediction-ml/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│ └── humidity_prediction.ipynb
│
├── src/
│ ├── model.py
│ 
│
├── dataset/
│ └── Met dataset - 2015-to-2022_12months.csv
│
├── images/
│ └── results.png
│
└── report/
└── report.pdf


---

# Data Preprocessing

The following preprocessing techniques were applied:

## Missing Value Handling

- Mean Imputation
- KNN Imputer

## Data Cleaning

- Removed duplicate values
- Removed invalid rows

## Feature Scaling

StandardScaler used:

X_scaled = (X − mean) / standard deviation


## Outlier Removal

Outliers detected using boxplots and removed.

## Data Splitting

Chronological time-series split:

- Training: 64%
- Validation: 16%
- Testing: 20%

---

# Feature Engineering

Time-shifted features created using 1-month lookback.

This allows prediction of future humidity based on previous month data.

Recursive Feature Elimination (RFE) used to select most relevant features.

---

# Model Architecture

Multi-Layer Perceptron Neural Network

Structure:

Input Layer

Hidden Layer 1

- 64 neurons
- ReLU activation
- Dropout 0.2

Hidden Layer 2

- 32 neurons
- ReLU activation
- Dropout 0.2

Hidden Layer 3

- 16 neurons
- ReLU activation

Output Layer

- 1 neuron
- Linear activation

---

# Overfitting Prevention Techniques

Multiple techniques used:

Early Stopping

Stops training when validation loss stops improving.

Dropout

Randomly disables neurons during training.

L2 Regularization

Penalizes large weights.

Feature Selection

Removes irrelevant features.

Validation Set

Used for hyperparameter tuning.

---

# Model Performance

Main Results:

Root Mean Squared Error:

4.64

Mean Absolute Error:

3.80

R-Squared Score:

0.038

Pearson Correlation:

0.853

---

Linear Regression Comparison:

RMSE:

1.72

R-Squared:

0.867

---

# Technologies Used

Python

TensorFlow

Keras

Scikit-Learn

NumPy

Pandas

Matplotlib

Jupyter Notebook

---

# Installation

Clone repository:

git clone [https://github.com/Innovator-Nick/humidity-prediction-ml.git]


Navigate to folder:

cd humidity-prediction-ml


Install dependencies:

pip install -r requirements.txt


---

# How to Run

Run Jupyter Notebook:

jupyter notebook notebooks/humidity_prediction.ipynb


Or run Python script:

python src/model.py


---

# Results

The model successfully captured patterns in humidity data.

Pearson correlation of 0.85 indicates strong prediction capability.

Neural network generalised well on unseen data.

---

# Future Improvements

Possible improvements:

Use LSTM (better for time series)

Hyperparameter tuning

Use larger dataset

Deploy as web app

---

# Author

Innovator-Nick

Machine Learning Student

GitHub:

https://github.com/Innovator-Nick

---

# License

MIT License

You are free to use this project.




