
# 📊 Financial Sentiment Analysis

## 📌 Overview
This project analyzes sentiment in **financial news and social media** using **Natural Language Processing (NLP) and Machine Learning**. The goal is to classify financial text into **positive, negative, or neutral** sentiments and explore its impact on stock market movements.

## 📂 Dataset
- **Source:** Financial news articles and stock-related tweets.
- **Format:** CSV files containing text and sentiment labels.
- **Example Features:**
  - **Text:** Financial news headlines or tweets.
  - **Sentiment Label:** Positive, Negative, or Neutral.

## 🚀 Methodology
1. **Data Preprocessing:**
   - Read CSV files using Pandas.
   - Remove stopwords, punctuation, and perform lemmatization.
   - Convert text to lowercase and clean special characters.
2. **Feature Engineering:**
   - Convert text into numerical features using **CountVectorizer** or **TF-IDF**.
   - Experiment with **Word Embeddings** (Word2Vec, BERT).
3. **Model Training:**
   - Applied **Machine Learning models** such as:
     - Logistic Regression
     - Random Forest
     - XGBoost
     - LSTMs (for deep learning approach)
   - Evaluated using **accuracy, precision, recall, and F1-score**.

## 📊 Results
- Achieved **high accuracy** in sentiment classification using optimized models.
- Identified a **correlation between sentiment polarity and stock price fluctuations**.

## 🔧 Technologies Used
- **Programming Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK, TensorFlow/Keras
- **Visualization:** Matplotlib, Seaborn

## 💎 Challenges & Solutions
- **Noisy Financial Text:** Applied **custom stopword filtering and domain-specific sentiment lexicons**.
- **Imbalanced Sentiment Classes:** Used **SMOTE and class weighting techniques**.


