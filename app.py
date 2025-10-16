import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# 1️⃣ Load Model, Columns & Vectorizer
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model1.pkl")
    vectorizer = joblib.load("scaler1.pkl")  # TF-IDF vectorizer
    columns = joblib.load("columns1.pkl")    # Not used for NLP, but we’ll load anyway
    return model, vectorizer, columns

model, vectorizer, columns = load_artifacts()

# ---------------------------
# 2️⃣ Streamlit UI
# ---------------------------
st.set_page_config(page_title="Emotion Prediction App", page_icon="🧠", layout="centered")

st.title("🧠 Emotion Prediction using NLP")
st.write("Enter any text below and find out the **emotion** behind it!")

# ---------------------------
# 3️⃣ User Input
# ---------------------------
user_text = st.text_area("💬 Type your sentence here:", height=150)

# ---------------------------
# 4️⃣ Predict Emotion
# ---------------------------
if st.button("🔍 Predict Emotion"):
    if not user_text.strip():
        st.warning("Please enter some text first!")
    else:
        # Transform input text using TF-IDF vectorizer
        X_input = vectorizer.transform([user_text])

        # Predict emotion
        prediction = model.predict(X_input)[0]
        probabilities = model.predict_proba(X_input)[0]

        # Get emotion labels from model
        emotions = model.classes_

        # Create probability DataFrame
        prob_df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": np.round(probabilities, 3)
        }).sort_values(by="Probability", ascending=False)

        # Display Results
        st.subheader("🎯 Predicted Emotion:")
        st.success(f"**{prediction.upper()}**")

        st.subheader("📊 Emotion Probabilities:")
        st.bar_chart(prob_df.set_index("Emotion"))

# ---------------------------
# 5️⃣ Footer
# ---------------------------
st.markdown("---")
st.markdown("Made with ❤️ by Piyush | NLP Emotion Detection Demo")
