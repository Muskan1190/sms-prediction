# 📩 SMS Spam Classifier

This project builds a machine learning pipeline to classify SMS messages as **spam** or **ham** using the [UCI SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). The goal is to compare different classification models using both TF-IDF and Bag of Words techniques to determine the best-performing approach.

---

## 📊 Dataset Overview

- **Total messages:** 5,574
- **Labels:** `spam`, `ham`
- **Features:** SMS message content

Each row represents an SMS message with a label indicating whether it's spam or not.

---

## 🧹 Preprocessing

The following steps were used to clean and prepare the text:
- Lowercasing text
- Removing punctuation and special characters
- Tokenization
- Removing stopwords
- Lemmatization (using `nltk`)

---

## ✨ Feature Extraction

Two feature extraction methods were explored:

- **Bag of Words (BoW)** – Simple frequency-based vectorization
- **TF-IDF** – Weights terms by their importance in the document and across the corpus

---

## 🤖 Models Used

Four classifiers were trained and evaluated:

1. **Random Forest Classifier**

2. **Gradient Boosting Classifier**

3. **XGBoost Classifier**

4. **Multinomial Naive Bayes**

---

## 🧪 Evaluation Metrics

Each model was evaluated using:
- **Accuracy**
- **Precision**
- **Confusion Matrix**
- **Word cloud visualizations**

---

## 📈 Results Summary

| Model                      | Accuracy | Precision |
|---------------------------|----------|-----------|
| **Random Forest (RF)**     | **0.9739**  | **1.0000**   |
| Gradient Boosting (GBC)    | 0.9574   | 0.9123    |
| XGBoost (XGB)              | 0.9671   | 0.9561    |
| Multinomial Naive Bayes (MN)| 0.9594   | 1.0000    |

---

## 📌 Key Insights

- ✅ **Random Forest** delivered the best overall performance with **perfect precision** and the highest accuracy.
- 📈 **Naive Bayes** performed surprisingly well with **perfect precision**, despite its simplicity.
- 🔍 **XGBoost** showed reliable, balanced performance across both metrics.
- ⚠️ **GBC** was slightly less precise, occasionally misclassifying ham messages as spam.

---

## 🛠️ Tech Stack

- `pandas`, `numpy` for data manipulation
- `nltk` for text preprocessing
- `scikit-learn`, `xgboost` for modeling
- `matplotlib`, `seaborn`, `wordcloud` for visualization

---

## 💡 Future Work

- Tune hyperparameters using `GridSearchCV` or `Optuna`
- Try deep learning approaches (LSTM, BERT)
- Deploy using Streamlit for real-time predictions

---

## 📁 Dataset

📂 [Kaggle: SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 👩‍💻 Author

**Muskan**  
Passionate about machine learning and solving real-world problems.  
Let's connect and collaborate!

