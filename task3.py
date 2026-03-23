#Importing Libraries
import pandas as pd
import pandas as pd
import numpy as np
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Loading Dataset
data=pd.read_csv('Resume.csv')
df=pd.DataFrame(data)
df.head()
job_description = "Looking for data analyst with Python SQL Excel and data visualization skills"
#Text Cleaning
def clean_text(text):
    if not isinstance(text, str): # Handle potential empty/NaN rows
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
df["clean_resume"] = df["Resume_str"].apply(clean_text)
job_clean = clean_text(job_description)
#TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
all_docs = [job_clean] + df["clean_resume"].tolist()
tfidf_matrix = tfidf.fit_transform(all_docs)
#Cosine Similarity
scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
df['match_score'] = scores[0] * 100
#Skill Extraction
def find_missing_skills(resume, jd):
    jd_words = set(jd.split())
    res_words = set(resume.split())
    missing = jd_words - res_words
    # Return missing words as a string, or "None" if they have everything
    return ", ".join(list(missing)) if missing else "Perfect Match"

df['missing_skills'] = df['clean_resume'].apply(lambda x: find_missing_skills(x, job_clean))
df_ranked = df.sort_values(by='match_score', ascending=False)
print("--- Top 5 Candidates Found ---")
print(df_ranked[['Resume_str', 'match_score', 'missing_skills']].head(5))
#This system evaluates resumes by comparing them with a job description using NLP techniques. Candidates are ranked based on similarity scores, and missing skills are highlighted to provide actionable insights. This helps streamline the screening process and enables faster, data-driven hiring decisions.

