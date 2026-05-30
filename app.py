import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("FAQ Chatbot")

data = pd.read_csv("faq.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

questions = data["Question"].apply(clean_text)

user_question = st.text_input("Ask your question")

if user_question:

    user_question = clean_text(user_question)

    all_questions = list(questions)
    all_questions.append(user_question)

    tfidf = TfidfVectorizer()

    vectors = tfidf.fit_transform(all_questions)

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )

    best_match = similarity.argmax()

    score = similarity.max()

    if score > 0.2:
        st.write("Answer:")
        st.success(data.iloc[best_match]["Answer"])
        st.write("Match Score:", round(score * 100, 2), "%")
    else:
        st.error("Sorry, answer not found.")