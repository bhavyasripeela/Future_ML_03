import streamlit as st
import pandas as pd
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("AI Resume Screening System")

st.write("Upload a job description and rank resumes based on relevance.")

# Load dataset
df = pd.read_csv("Resume.csv")

# Make sure column exists
df['Resume_str'] = df['Resume_str'].astype(str)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['clean_resume'] = df['Resume_str'].apply(clean_text)

# Input job description
job_description = st.text_area("Enter Job Description")

if st.button("Analyze Resumes"):
    
    if job_description.strip() == "":
        st.warning("Please enter a job description")
    else:
        job_clean = clean_text(job_description)

        # Combine resumes + job
        all_text = df['clean_resume'].tolist() + [job_clean]

        # TF-IDF
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(all_text)

        # Similarity
        similarity = cosine_similarity(vectors[:-1], vectors[-1])

        df['match_score'] = similarity

        # Rank
        df_ranked = df.sort_values(by='match_score', ascending=False)

        # Display
        st.subheader("Top Candidates")
        st.dataframe(df_ranked[['Resume_str', 'match_score']].head(5))
