
import streamlit as st
import torch
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# APP SETTINGS
# =========================
st.set_page_config(page_title="Urdu Sentiment Analyzer")

st.title("🧠 Urdu Sentiment Analysis App")
st.write("Enter Urdu text (sentence or paragraph) to analyze sentiment.")

# =========================
# LOAD MODEL FROM HUGGING FACE
# =========================
@st.cache_resource
def load_model():

    model_name = "abd12-tahir/urdu-sentiment-model"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_model()

# =========================
# DEVICE SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# =========================
# SPLIT URDU SENTENCES
# =========================
def split_urdu_sentences(text):
    sentences = re.split(r"[۔?!]", text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]

# =========================
# PREDICT SENTIMENT
# =========================
def predict_sentiment(sentence):

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=-1)

    confidence, predicted_class = torch.max(probabilities, dim=1)
    label = model.config.id2label[predicted_class.item()]

    return label, float(confidence)

# =========================
# USER INPUT
# =========================
user_input = st.text_area("✍️ Enter Urdu text:", height=150)

# =========================
# BUTTON ACTION
# =========================
if st.button("Analyze Sentiment"):

    if not user_input.strip():
        st.warning("⚠️ Please enter Urdu text")
    else:
        sentences = split_urdu_sentences(user_input)

        st.subheader("📊 Sentence-wise Results")

        results = []

        for sentence in sentences:
            label, confidence = predict_sentiment(sentence)
            results.append(label)

            if label == "positive":
                st.success(f"{sentence} → ✅ Positive ({confidence:.2f})")

            elif label == "negative":
                st.error(f"{sentence} → ❌ Negative ({confidence:.2f})")

            else:
                st.info(f"{sentence} → ⚖️ Neutral ({confidence:.2f})")

        # =========================
        # OVERALL SENTIMENT
        # =========================
        if results:
            overall = max(set(results), key=results.count)
            st.subheader(f"⭐ Overall Sentiment: {overall.upper()}")
