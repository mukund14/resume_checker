import streamlit as st
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import PyPDF2
import io

client = OpenAI(api_key="your OpenAPI key goes here")

st.title("📄 ATS Resume Checker")

st.write("Upload your resume as a .docx or .pdf and paste a job description to see if you'd pass an ATS screen.")

uploaded_file = st.file_uploader("Upload your resume (.docx or .pdf)", type=["pdf", "docx"])
job_desc = st.text_area("Paste job description here")

resume = ""
if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            resume += page.extract_text()
    elif uploaded_file.name.endswith(".docx"):
        resume = docx2txt.process(uploaded_file)

if st.button("Check ATS Match") and resume and job_desc:
    # Cosine similarity
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, job_desc])
    cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

    # OpenAI Evaluation
    prompt = f"Evaluate the following resume against this job description. Return a score (0-100), a brief explanation, and recommendations for improving the resume if it doesn't meet the requirements.\n\nResume:\n{resume}\n\nJob Description:\n{job_desc}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    evaluation = response.choices[0].message.content
    passed = cos_sim >= 60

    st.subheader("Results")
    st.write(f"Cosine Similarity Score: {cos_sim:.2f} %")
    st.markdown("**GPT-4 Evaluation:**")
    st.write(evaluation)
    st.markdown(f"### {'✅ PASS' if passed else '❌ FAIL'}")
