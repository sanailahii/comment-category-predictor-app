# Multi-Class Comment Classification using NLP

A machine learning application that classifies user comments into four categories:

* Normal
* Identity
* Toxic
* Extreme

## Project Overview

This project was developed as part of a machine learning classification task involving large-scale text data.

The solution combines:

* TF-IDF Vectorization
* Logistic Regression
* Feature Scaling
* Text and Numerical Feature Engineering

## Model Details

* Vectorizer: TF-IDF (70,000 features)
* Classifier: Logistic Regression
* Hyperparameter: C = 2
* Custom Class Weights for Imbalanced Data

## Features Used

### Text Features

* User Comment

### Numerical Features

* Upvotes
* Downvotes
* Emoticon 1
* Emoticon 2
* Emoticon 3

For deployment, default values are used for numerical features to simplify user interaction.

## Technologies Used

* Python
* Scikit-Learn
* Pandas
* NumPy
* Streamlit
* TF-IDF
* Logistic Regression

## Model Training Repository

The complete training pipeline, EDA, and experiments are available in:
https://github.com/sanailahii/comment-category-predictor-nlp

## Deployment

The application predicts comment categories and displays prediction confidence scores for all classes.
Note: This application uses a TF-IDF + Logistic Regression model. Predictions are based on learned textual patterns and may not fully capture contextual nuances such as negation ("don't kill") that modern transformer-based models handle more effectively.

## Author

Sana Ilahi
