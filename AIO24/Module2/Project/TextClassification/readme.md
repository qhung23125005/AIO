# Project: Spam Message Classification with Naive Bayes

## Overview

This project is part of **AIO2024 Module 02** and focuses on **text classification**, specifically classifying messages as **spam or ham (not spam)** using the **Naive Bayes algorithm**. The system is designed to analyze the content of messages and determine whether they are spam based on predefined patterns.

## Features

- **Binary Classification**: Classifies messages into **Spam** or **Ham**.
- **Preprocessing Pipeline**:
  - Tokenization
  - Stopword removal
  - Stemming
  - Feature extraction using word frequency
- **Naive Bayes Classifier**: Implements **Gaussian Naive Bayes** for classification.
- **Dataset Handling**: Uses a labeled dataset with spam and ham messages.
- **Performance Evaluation**: Computes accuracy, precision, and recall metrics.

## Technologies Used
- **Python** (NumPy, Pandas, Matplotlib)
- **Machine Learning**: Scikit-learn (Naive Bayes classifier)
- **Natural Language Processing (NLP)**: NLTK for text preprocessing

## Dataset

- The dataset contains two columns:
  - **Category**: Labels (`spam` or `ham`).
  - **Message**: Text content of messages.
- Preprocessing includes **lowercasing, punctuation removal, tokenization, and stopword filtering**.

## Results
- Evaluates the model’s performance using validation and test datasets.


