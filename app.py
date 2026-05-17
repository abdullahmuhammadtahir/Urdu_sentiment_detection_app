import streamlit as st
import torch
import re
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Urdu Sentiment Analyzer", layout="centered")

st.title("🧠 Urdu Sentiment Analysis App")
st.write("Analyze Urdu text for Positive, Negative, or Neutral sentiment.")

# =========================
# LOAD MODEL (Hugging Face)
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
# SPLIT SENTENCES
# =========================
def split_urdu_sentences(text):
    sentences = re.split(r"[۔?!]", text)
    return [s.strip() for s in sentences if len(s.strip()) > 2]

# =========================
# ADVANCED PREDICTION ✅
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
        probs = F.softmax(outputs.logits, dim=-1)

    probs = probs[0]

    # get all class probabilities
    label_map = model.config.id2label
    prob_dict = {label_map[i]: probs[i].item() for i in range(len(probs))}

    # get best label
    best_label = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[best_label]

    # ✅ FIX 1: LOW CONFIDENCE → NEUTRAL
    if confidence < 0.60:
        best_label = "neutral"

    # ✅ FIX 2: STRONG NEGATIVE RECOVERY
    if prob_dict["negative"] > 0.35 and prob_dict["negative"] >= prob_dict[best_label] - 0.1:
        best_label = "negative"

    # ✅ FIX 3: STRONG POSITIVE CHECK
    if prob_dict["positive"] > 0.65:
        best_label = "positive"

    return best_label, confidence

# =========================
# USER INPUT
# =========================
user_input = st.text_area("✍️ Enter Urdu text:", height=150)

# =========================
# ANALYSIS BUTTON
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
        # OVERALL RESULT
        # =========================
        if results:
            overall = max(set(results), key=results.count)
            st.subheader(f"⭐ Overall Sentiment: {overall.upper()}")
``

