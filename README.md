# 💊 Drug Review Sentiment Analysis: NLP and Machine Learning Pipeline

This project performs sentiment classification on drug reviews using Natural Language Processing (NLP) techniques and machine learning models. Based on real-world user reviews from DrugLib.com, the pipeline demonstrates how unstructured text can be processed, analyzed, and transformed into predictive insights within the healthcare domain.

---

## 🎯 Objective

To build an end-to-end pipeline that predicts user sentiment (positive/negative) based on their written drug reviews. This helps highlight how patients perceive medication effectiveness, aiding pharmaceutical insights.

---

## 🔁 Pipeline Overview

### ✅ 1. Data Preprocessing
- Lowercasing, punctuation & digit removal
- Tokenization using `nltk`
- Stopwords removal
- Stemming or Lemmatization (optional)

### ✅ 2. Feature Extraction
- TF-IDF Vectorization for text-to-numeric conversion

### ✅ 3. Model Training
- Logistic Regression
- Multinomial Naive Bayes
- Optional: Support Vector Machine (SVM), Random Forest

### ✅ 4. Model Evaluation
- Accuracy
- Confusion Matrix
- F1-Score
- ROC Curve (optional)

---

## 📊 Visualizations

- Word clouds for positive vs negative reviews
- Sentiment class distribution
- Important TF-IDF features

---

## 🧰 Tech Stack

- Python
- pandas, numpy
- nltk / spaCy
- scikit-learn
- matplotlib, seaborn, wordcloud

---

## 🚀 Potential Extensions

- Integrate deep learning with BERT via Hugging Face
- Convert pipeline into a REST API using Flask
- Build a frontend app with Streamlit for live sentiment prediction

---
