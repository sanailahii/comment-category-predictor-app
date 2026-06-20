import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack

# Load saved objects
model = joblib.load("comment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Multi-Class Comment Classification using NLP")
st.caption("Developed by Sana Ilahi")

st.write(
    "Predict whether a comment belongs to Normal, Identity, Toxic, or Extreme categories "
    "using a machine learning model trained on text data."
)

comment = st.text_area("Enter a comment")

if st.button("Predict"):

    # Default numeric values
    num_df = pd.DataFrame({
        "upvote": [0],
        "downvote": [0],
        "emoticon_1": [0],
        "emoticon_2": [0],
        "emoticon_3": [0]
    })

    text_features = tfidf.transform([comment])
    num_features = scaler.transform(num_df)

    features = hstack([text_features, num_features])

    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]

    labels = {
        0: "Normal",
        1: "Identity",
        2: "Toxic",
        3: "Extreme"
    }

    st.success(f"Predicted Category: {labels[prediction]}")

    st.subheader("Prediction Confidence")

    for i, prob in enumerate(probs):
        st.progress(float(prob))
        st.write(f"{labels[i]}: {prob:.2%}")

st.markdown("### Example Comments")

st.write("- I love this community")
st.write("- You are an idiot")
st.write("- Kill them all")
st.write("- I completely disagree with your opinion")