import streamlit as st
from transformers import pipeline
from langchain_experimental.text_splitter import ExperimentalTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# Title of the application
st.title("Sentiment Analysis with Semantic Chunking")

# Sidebar for user input options
st.sidebar.header("Model Options")

# Select model from a predefined list of sentiment analysis models
model_name = st.sidebar.selectbox(
    "Choose a Hugging Face Sentiment Model:",
    ["distilbert-base-uncased-finetuned-sst-2-english", "bert-base-uncased-finetuned-sst-2"]
)

# Load the selected model
@st.cache_resource
def load_model(model_name):
    return pipeline("sentiment-analysis", model=model_name)

model_pipeline = load_model(model_name)

# Main input text box
st.subheader("Input Text")
user_input = st.text_area("Enter text to analyze:", "The movie was fantastic!")

# Sidebar option for chunk size
chunk_size = st.sidebar.slider("Chunk Size (characters):", min_value=100, max_value=1000, value=500, step=50)

# Initialize OpenAI embeddings
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

embeddings = get_embeddings()

# Function to perform semantic chunking using experimental chunker
def semantic_chunking(text, chunk_size, embeddings):
    splitter = ExperimentalTextSplitter(
        embeddings=embeddings,
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    return splitter.split_text(text)

if st.button("Run Sentiment Analysis"):
    with st.spinner("Running analysis..."):
        # Perform semantic chunking
        chunks = semantic_chunking(user_input, chunk_size, embeddings)

        st.subheader("Text Chunks")
        for i, chunk in enumerate(chunks):
            st.write(f"**Chunk {i + 1}:** {chunk}")

        # Run sentiment analysis on each chunk
        st.subheader("Sentiment Analysis Results")
        results = [model_pipeline(chunk) for chunk in chunks]
        for i, result in enumerate(results):
            st.write(f"**Chunk {i + 1} Sentiment:** {result}")

st.sidebar.markdown(
    "This application uses Hugging Face Transformers library for sentiment analysis and LangChain Experimental for semantic chunking with OpenAI embeddings."
)