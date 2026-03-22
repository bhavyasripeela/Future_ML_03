import streamlit as st
import pandas as pd
import string
import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("AI Resume Screening System")

st.write("Enter a job description to rank resumes")

# Load dataset
df = pd.read_csv("Resume.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df["clean_resume"] = df["Resume_str"].astype(str).apply(clean_text)

job_description = st.text_area("Enter Job Description")

if st.button("Analyze"):

    if job_description.strip() == "":
        st.warning("Please enter a job description")
    else:
        job_clean = clean_text(job_description)

        all_text = df["clean_resume"].tolist() + [job_clean]

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(all_text)

        scores = cosine_similarity(vectors[:-1], vectors[-1])

        df["match_score"] = scores

        df_ranked = df.sort_values(by="match_score", ascending=False)

        st.subheader("Top Candidates")
        st.dataframe(df_ranked[["Resume_str", "match_score"]].head(5))
