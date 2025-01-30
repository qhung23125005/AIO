# Sentiment Analysis Project

## **Overview**
This is a small project focusing on **sentiment analysis** using machine learning techniques. The goal is to apply knowledge which was learnt in this module to classify text as **positive, negative, or neutral** based on its sentiment. 

The pipeline includes:
- **Data Preprocessing** (Cleaning and transforming text)
- **Feature Engineering** (Vectorization)
- **Machine Learning Models** (Decision Tree & Random Forest)

---

## **1. Data Preprocessing**
Before training models, raw text data undergoes **preprocessing** to improve model performance. The following steps are performed:
- **Removing HTML Tags**: Extracting clean text from raw HTML.
- **Expanding Contractions**: Converting shortened words to their full forms.
- **Removing Emojis & URLs**: Eliminating unnecessary symbols and links.
- **Lowercasing & Punctuation Removal**: Standardizing text format.
- **Stopword Removal**: Filtering out common words that do not contribute to sentiment.
- **Lemmatization**: Converting words to their base form for consistency.

---

## **2. Feature Engineering**
To convert text into a format suitable for machine learning, **TF-IDF (Term Frequency-Inverse Document Frequency)** was applied. It assigns importance to words based on their occurrence in a document relative to the entire dataset.

---

## **3. Machine Learning Models**
This project utilizes basic supervised learning models for sentiment classification:
### **Decision Tree**
- A tree-based model that splits text features into decision nodes.
- Works well for interpretable models but may overfit without pruning.

### **Random Forest**
- An ensemble model consisting of multiple decision trees.
- Improves accuracy by reducing variance and preventing overfitting.

Both models are evaluated using standard classification metrics.

---

## **4. Evaluation Metrics**
The performance of sentiment analysis models is assessed using:
- **Accuracy**: Measures overall correctness of predictions.
