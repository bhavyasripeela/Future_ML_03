# 📄 AI Resume Screening & Ranking System

## 📌 Project Overview
This project builds a Machine Learning-based system to automatically **screen, score, and rank resumes** based on a given job description.

The system uses Natural Language Processing (NLP) techniques to compare resumes with job requirements and identify the best candidates, helping recruiters make faster and more informed hiring decisions.

---

## 🎯 Objective
The objective of this project is to:

- Automatically analyze resume text  
- Match candidate skills with job requirements  
- Rank candidates based on relevance  
- Identify missing skills in resumes  
- Support data-driven hiring decisions  

---

## 🛠 Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- Jupyter Notebook  

---

## ⚙️ How It Works

### 1️⃣ Data Preprocessing
- Convert text to lowercase  
- Remove punctuation  
- Clean and standardize resume text  

### 2️⃣ Feature Extraction
- Convert text into numerical form using:
  - TF-IDF Vectorization  

### 3️⃣ Similarity Calculation
- Compare resumes with job description using:
  - Cosine Similarity  

### 4️⃣ Candidate Ranking
- Rank candidates based on similarity scores  
- Higher score indicates better match  

### 5️⃣ Skill Gap Analysis
- Identify missing skills by comparing:
  - job description keywords  
  - resume content  

---

## 📊 Output
The system provides:

- Ranked list of candidates  
- Match score for each resume  
- Missing skills for each candidate  

---

## 💼 Business Impact
This system helps organizations:

- Reduce manual resume screening time  
- Improve candidate selection accuracy  
- Identify skill gaps quickly  
- Enhance hiring efficiency  

---

## 🚀 Streamlit App
This project is deployed using Streamlit, allowing users to:

- Enter a job description  
- Analyze resumes  
- View ranked candidates instantly  

---

## 📂 Project Structure
resume-screening-system

- app.py
- Resume.csv
- requirements.txt
- resume_screening.ipynb
- README.md
## ▶️ How to Run Locally
pip install -r requirements.txt 
streamlit run app.py
---

## 🌐 Live Demo
(Add your Streamlit app link here)

