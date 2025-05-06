import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from textblob import TextBlob

st.set_page_config(page_title="FashionReview AI", layout="centered")
st.title("🧥 FashionReview AI - Sentiment & Summary Tool")

st.markdown("""
Upload or paste customer reviews of fashion products. 
We'll analyze the sentiment and summarize them into a marketing-ready caption.
""")

llm = Ollama(model="tinyllama")

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""You are a fashion review assistant. Given the following customer reviews:

{text}

Summarize them in 2-3 lines and suggest a catchy marketing caption that highlights their sentiments.
"""
)

chain = LLMChain(llm=llm, prompt=prompt_template)

review_input = st.text_area("✍️ Paste Fashion Product Reviews:", height=200, placeholder="E.g. Love the jacket! Super warm and stylish. The zipper was a bit tight though.")

if st.button("🚀 Analyze & Summarize"):
    if review_input:
        st.subheader("🧠 Sentiment Classification")
        reviews = review_input.strip().split("\n")

        sentiment_results = []
        for review in reviews:
            blob = TextBlob(review)
            sentiment = blob.sentiment.polarity
            label = "Positive" if sentiment > 0.1 else ("Negative" if sentiment < -0.1 else "Neutral")
            sentiment_results.append((review, label))

        for i, (review, sentiment) in enumerate(sentiment_results):
            st.markdown(f"**Review {i+1}:** {review}")
            st.markdown(f"Sentiment: `{sentiment}`\n")

        with st.spinner("Generating summary and caption..."):
            output = chain.run(review_input)

        st.divider()
        st.subheader("📝 AI Summary & Caption")
        st.text_area("Generated Summary:", output, height=200)
    else:
        st.warning("Please paste some reviews to analyze.")
