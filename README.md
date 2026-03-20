# 🎬 AI Sentiment Studio

An interactive **Movie Review Sentiment Analysis Web App** built using **Natural Language Processing (NLP)** and **Machine Learning**, deployed with **Streamlit**.

The application predicts whether a movie review is **Positive 😊** or **Negative 😡**, and also explains the prediction by highlighting the most influential words.

---

## 🚀 Features

✅ Real-time sentiment prediction
✅ Clean modern dark UI
✅ Confidence score visualization
✅ Model explainability (important words display)
✅ NLP preprocessing (lemmatization, stopword removal, negation handling)
✅ TF-IDF feature extraction
✅ Machine Learning classification model
✅ Streamlit web deployment

---

## 🧠 Machine Learning Workflow

1. Text preprocessing

   * Lowercasing
   * Removing special characters
   * Stopword removal
   * Lemmatization
   * Negation handling

2. Feature Engineering

   * TF-IDF Vectorization
   * N-grams based representation

3. Model Training

   * Supervised ML classification model (e.g., Logistic Regression / Linear SVM)

4. Model Deployment

   * Model saved using **Pickle**
   * Integrated into Streamlit interface

---

## 📂 Project Structure

```
AI-Sentiment-Studio/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── Review_sentiment.ipynb
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ThePrinceM/Movie_Review_Sentiment_Analysis.git
cd Movie_Review_Sentiment_Analysis
```

### 2️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🌐 Deployment

This project can be deployed easily on:

* Streamlit Cloud
* Heroku
* Render
* AWS / GCP / Azure

---

## 📊 Example

**Input Review:**

> "The movie was not that bad and acting was good."

**Output:**

* Predicted Sentiment → Positive
* Confidence Score → 84%
* Influential Words → *good, not_bad*

---

## 🎯 Future Improvements

* Deep Learning model (LSTM / BERT)
* Real probability prediction
* Word highlighting inside full review
* Multi-language sentiment support
* User authentication & review history

---

## 👨‍💻 Author

Developed as part of an **NLP Machine Learning Project**.

---

⭐ If you like this project, consider giving it a **star on GitHub!**
