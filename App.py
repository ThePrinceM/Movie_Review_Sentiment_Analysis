import streamlit as st
import pickle
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ---------------- LOAD ----------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="AI Sentiment Studio", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg,#020617,#0f172a);
}

.header-glass {
    background: rgba(255,255,255,0.06);
    padding: 45px;
    border-radius: 24px;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 30px;
    text-align: center;
}

.header-title {
    font-size: 46px;
    font-weight: 700;
    color: white;
}

.header-subtitle {
    color: #cbd5e1;
    margin-top: 10px;
    font-size: 18px;
}

textarea {
    background:#020617 !important;
    color:white !important;
}

.stButton>button {
    background: linear-gradient(90deg,#06b6d4,#3b82f6);
    color:white;
    border:none;
    padding:12px;
    border-radius:10px;
    width:100%;
    font-size:16px;
    transition:0.3s;
}

.stButton>button:hover {
    transform: scale(1.04);
}

.sentiment {
    text-align:center;
    font-size:28px;
    font-weight:600;
    padding:18px;
    border-radius:14px;
    margin-top:20px;
}

.pos {
    background: linear-gradient(90deg,#16a34a,#22c55e);
    color:white;
}

.neg {
    background: linear-gradient(90deg,#dc2626,#ef4444);
    color:white;
}

.word-chip {
    display:inline-block;
    padding:8px 14px;
    margin:6px;
    border-radius:20px;
    font-size:14px;
}

.pos-chip {
    background:#16a34a;
    color:white;
}

.neg-chip {
    background:#dc2626;
    color:white;
}

.footer {
    text-align:center;
    margin-top:35px;
    color:#94a3b8;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-glass">
    <div class="header-title">AI Sentiment Studio</div>
    <div class="header-subtitle">Understand emotions behind movie reviews</div>
</div>
""", unsafe_allow_html=True)

# ---------------- PREPROCESS ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)

    tokens = text.split()

    new_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "not" and i+1 < len(tokens):
            new_tokens.append("not_" + tokens[i+1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    tokens = [w for w in new_tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# ---------------- EXPLAIN ----------------
def explain_prediction(text):
    feature_names = np.array(vectorizer.get_feature_names_out())
    vector = vectorizer.transform([text])

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        indices = vector.nonzero()[1]

        important = sorted(
            [(feature_names[i], coefs[i]) for i in indices],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:6]

        return important
    return []

# ---------------- INPUT AREA ----------------
review = st.text_area("Write your movie review")

# ---------------- PREDICT ----------------
if st.button("Analyze Sentiment"):

    cleaned = preprocess(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if hasattr(model,"decision_function"):
        score = abs(model.decision_function(vector)[0])
        confidence = min(100, int(score*20))
    else:
        confidence = 75

    if prediction == "positive":
        st.markdown('<div class="sentiment pos">😊 Positive Sentiment</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sentiment neg">😡 Negative Sentiment</div>', unsafe_allow_html=True)

    st.progress(confidence/100)
    st.caption(f"Confidence Score: {confidence}%")

    st.markdown("### Influential Words")

    words = explain_prediction(cleaned)

    for word, weight in words:
        if weight > 0:
            st.markdown(f'<span class="word-chip pos-chip">{word}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="word-chip neg-chip">{word}</span>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">NLP Sentiment Analysis Project | Streamlit</div>', unsafe_allow_html=True)