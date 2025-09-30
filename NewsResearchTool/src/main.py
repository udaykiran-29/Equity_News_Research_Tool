# import os
# import streamlit as st
# import pickle
# import time
# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# from datetime import datetime
# # from langchain import HuggingFaceHub
# from langchain_community.llms import HuggingFaceHub
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv
# import plotly.express as px

# # Load environment variables
# load_dotenv()

# # Check if API key exists
# if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     st.error("Please set your Hugging Face API token in the .env file!")
#     st.stop()

# def extract_article_info(url):
#     """Extract article information using BeautifulSoup"""
#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         # Basic extraction of title and text
#         title = soup.title.string if soup.title else "No title found"
        
#         # Get all paragraphs
#         paragraphs = soup.find_all('p')
#         text = ' '.join([p.text for p in paragraphs])
        
#         # Create a simple summary (first 200 characters)
#         summary = text[:200] + "..." if len(text) > 200 else text
        
#         return {
#             'title': title,
#             'authors': 'Not available',
#             'publish_date': 'Not available',
#             'summary': summary,
#             'keywords': 'Not available'
#         }
#     except Exception as e:
#         st.warning(f"Could not extract information from {url}: {str(e)}")
#         return None

# def search_news(query, num_results=5):
#     """Search news articles using NewsAPI"""
#     api_key = os.getenv("NEWS_API_KEY")
    
#     if not api_key:
#         st.warning("NewsAPI key not found. Please add it to your .env file.")
#         return []
    
#     url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize={num_results}&language=en&sortBy=relevancy"
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             return response.json()['articles']
#         else:
#             st.warning(f"NewsAPI returned status code: {response.status_code}")
#             return []
#     except Exception as e:
#         st.error(f"Error fetching news: {str(e)}")
#         return []

# # Streamlit interface
# st.set_page_config(page_title="Equity News Research Tool", page_icon="💹", layout="wide")

# # Sidebar configuration
# st.sidebar.title("📈 Equity News Research")
# analysis_mode = st.sidebar.selectbox(
#     "Choose Analysis Mode",
#     ["Manual URL Input", "Search By Topic"]
# )

# # Initialize session state
# if 'urls' not in st.session_state:
#     st.session_state.urls = []
# if 'processed_data' not in st.session_state:
#     st.session_state.processed_data = None
# if 'article_summaries' not in st.session_state:
#     st.session_state.article_summaries = {}

# # Main content
# st.title("📈Equity News Research Tool")

# if analysis_mode == "Manual URL Input":
#     urls_input = st.text_area(
#         "Enter URLs (one per line)",
#         height=150,
#         help="Paste multiple URLs, each on a new line"
#     )
    
#     if urls_input:
#         st.session_state.urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

# elif analysis_mode == "Search By Topic":
#     search_query = st.text_input("Enter Topic name to search:")
#     if search_query:
#         with st.spinner("Searching for relevant articles..."):
#             articles = search_news(search_query)
#             if articles:
#                 st.success(f"Found {len(articles)} relevant articles!")
#                 selected_articles = st.multiselect(
#                     "Select articles to analyze:",
#                     options=[article['title'] for article in articles],
#                     default=[article['title'] for article in articles[:3]]
#                 )
#                 st.session_state.urls = [
#                     article['url'] for article in articles
#                     if article['title'] in selected_articles
#                 ]

# # Process URLs button
# if st.session_state.urls:
#     st.subheader("Selected Articles")
#     for url in st.session_state.urls:
#         with st.expander(f"Article Details - {url}"):
#             info = extract_article_info(url)
#             if info:
#                 st.write(f"**Title:** {info['title']}")
#                 st.write("**Summary:**")
#                 st.write(info['summary'])
#                 st.session_state.article_summaries[url] = info

#     process_clicked = st.button("Analyze Articles")
    
#     if process_clicked:
#         main_placeholder = st.empty()
        
#         # Process URLs
#         with st.spinner("Processing articles..."):
#             loader = UnstructuredURLLoader(urls=st.session_state.urls)
#             main_placeholder.text("Loading articles... ⏳")
#             data = loader.load()
            
#             text_splitter = RecursiveCharacterTextSplitter(
#                 separators=['\n\n', '\n', '.', ','],
#                 chunk_size= 1000
#             )
#             main_placeholder.text("Processing text... ⏳")
#             docs = text_splitter.split_documents(data)
            
#             embeddings = HuggingFaceEmbeddings(
#                 model_name="sentence-transformers/all-mpnet-base-v2"
#             )
#             vectorstore_hf = FAISS.from_documents(docs, embeddings)
#             main_placeholder.text("Building knowledge base... ⏳")
            
#             # Save processed data
#             st.session_state.processed_data = vectorstore_hf
#             main_placeholder.empty()
#             st.success("Articles processed successfully! You can now ask questions.")

# # Question and Analysis Section
# if st.session_state.processed_data:
#     st.subheader("📝 Ask Questions About the Articles")
    
#     # Predefined questions
#     suggested_questions = [
#         "What are the main points discussed in these articles?",
#         "What are the key findings or conclusions?",
#         "What are the potential implications?",
#         "Compare and contrast the different viewpoints presented.",
#         "What evidence is provided to support the main arguments?"
#     ]
    
#     question_type = st.radio(
#         "Choose question type:",
#         ["Custom Question", "Suggested Questions"]
#     )
    
#     if question_type == "Suggested Questions":
#         question = st.selectbox("Select a question:", suggested_questions)
#     else:
#         question = st.text_input("Enter your question:")
    
#     if question:
#         with st.spinner("Analyzing..."):
#             # llm = HuggingFaceHub(
#             #     repo_id="google/flan-t5-large",
#             #     model_kwargs={"temperature": 0.9, "max_length": 512}
#             # )
#             llm = HuggingFaceHub(
#                 repo_id="tiiuae/falcon-7b-instruct",   # smaller model works with free API
#                 task="text2text-generation",
#                 huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
#                 model_kwargs={"temperature": 0.7, "max_length": 512}
#             )

            
#             chain = RetrievalQAWithSourcesChain.from_llm(
#                 llm=llm,
#                 retriever=st.session_state.processed_data.as_retriever()
#             )
            
#             result = chain({"question": question}, return_only_outputs=True)
            
#             # Display results
#             with st.container():
#                 st.markdown("### 📊 Analysis Results")
#                 st.write(result["answer"])
                
#                 if result.get("sources"):
#                     st.markdown("#### 📚 Sources")
#                     sources_list = result["sources"].split("\n")
#                     for source in sources_list:
#                         if source.strip():
#                             st.markdown(f"- {source}")

# # Download section
# if st.session_state.processed_data and st.session_state.article_summaries:
#     st.subheader("📥 Download Analysis")
    
#     # Prepare summary data
#     summary_data = []
#     for url, info in st.session_state.article_summaries.items():
#         summary_data.append({
#             "URL": url,
#             "Title": info['title'],
#             "Summary": info['summary']
#         })
    
#     df = pd.DataFrame(summary_data)
    
#     # Download button
#     csv = df.to_csv(index=False)
#     st.download_button(
#         label="Download Article Analysis (CSV)",
#         data=csv,
#         file_name="news_analysis.csv",
#         mime="text/csv"
#     )

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center'>
#         <p>Equity News Research Tool - Powered by Team 5🤖</p>
#         <p>Copyright © 2024 Team 5
#         . All rights reserved.</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )
import os
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables
load_dotenv()

# Check NewsAPI key
if not os.getenv("NEWS_API_KEY"):
    st.error("Please set your NEWS_API_KEY in the .env file!")
    st.stop()

# Streamlit config
st.set_page_config(page_title="Equity News Research Tool", page_icon="💹", layout="wide")
st.title("📈 Equity News Research Tool")

# Sidebar
st.sidebar.title("📈 Equity News Research")
analysis_mode = st.sidebar.selectbox("Choose Analysis Mode", ["Manual URL Input", "Search By Topic"])

# Initialize session state
if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'article_summaries' not in st.session_state:
    st.session_state.article_summaries = {}

# ------------------------
# Helper functions
# ------------------------

def extract_article_info(url):
    """Extract article information using BeautifulSoup"""
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "No title found"
        paragraphs = soup.find_all('p')
        text = ' '.join([p.text for p in paragraphs])
        summary = text[:200] + "..." if len(text) > 200 else text
        return {'title': title, 'authors': 'Not available', 'publish_date': 'Not available', 'summary': summary, 'keywords': 'Not available'}
    except Exception as e:
        st.warning(f"Could not extract information from {url}: {str(e)}")
        return None

def search_news(query, num_results=5):
    """Search news articles using NewsAPI"""
    api_key = os.getenv("NEWS_API_KEY")
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&pageSize={num_results}&language=en&sortBy=relevancy"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['articles']
        else:
            st.warning(f"NewsAPI returned status code: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

# ------------------------
# User Input
# ------------------------
if analysis_mode == "Manual URL Input":
    urls_input = st.text_area("Enter URLs (one per line)", height=150)
    if urls_input:
        st.session_state.urls = [url.strip() for url in urls_input.split('\n') if url.strip()]

elif analysis_mode == "Search By Topic":
    search_query = st.text_input("Enter Topic name to search:")
    if search_query:
        with st.spinner("Searching for relevant articles..."):
            articles = search_news(search_query)
            if articles:
                st.success(f"Found {len(articles)} relevant articles!")
                selected_articles = st.multiselect(
                    "Select articles to analyze:",
                    options=[article['title'] for article in articles],
                    default=[article['title'] for article in articles[:3]]
                )
                st.session_state.urls = [
                    article['url'] for article in articles if article['title'] in selected_articles
                ]

# ------------------------
# Process Articles
# ------------------------
if st.session_state.urls:
    st.subheader("Selected Articles")
    for url in st.session_state.urls:
        with st.expander(f"Article Details - {url}"):
            info = extract_article_info(url)
            if info:
                st.write(f"**Title:** {info['title']}")
                st.write("**Summary:**")
                st.write(info['summary'])
                st.session_state.article_summaries[url] = info

    process_clicked = st.button("Analyze Articles")
    
    if process_clicked:
        main_placeholder = st.empty()
        with st.spinner("Processing articles..."):
            loader = UnstructuredURLLoader(urls=st.session_state.urls)
            main_placeholder.text("Loading articles... ⏳")
            data = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
            main_placeholder.text("Processing text... ⏳")
            docs = text_splitter.split_documents(data)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vectorstore_hf = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Building knowledge base... ⏳")
            
            st.session_state.processed_data = vectorstore_hf
            main_placeholder.empty()
            st.success("Articles processed successfully! You can now ask questions.")

# ------------------------
# Question and Answer
# ------------------------
if st.session_state.processed_data:
    st.subheader("📝 Ask Questions About the Articles")

    suggested_questions = [
        "What are the main points discussed in these articles?",
        "What are the key findings or conclusions?",
        "What are the potential implications?",
        "Compare and contrast the different viewpoints presented.",
        "What evidence is provided to support the main arguments?"
    ]
    
    question_type = st.radio("Choose question type:", ["Custom Question", "Suggested Questions"])
    question = st.selectbox("Select a question:", suggested_questions) if question_type == "Suggested Questions" else st.text_input("Enter your question:")

    if question:
        with st.spinner("Analyzing..."):
            # Initialize Falcon model pipeline
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            model = AutoModelForCausalLM.from_pretrained("facebook/bart-large-cnn", device_map="auto", torch_dtype="auto")
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            # Retrieve context from vectorstore
            retriever = st.session_state.processed_data.as_retriever()
            docs = retriever.get_relevant_documents(question)
            context_text = " ".join([doc.page_content for doc in docs])
            
            # Combine question + context
            prompt = f"Context: {context_text}\n\nQuestion: {question}\nAnswer:"
            
            output = generator(prompt, max_length=512, do_sample=True, temperature=0.7)
            answer = output[0]['generated_text']
            
            st.markdown("### 📊 Analysis Results")
            st.write(answer)

# ------------------------
# Download Section
# ------------------------
if st.session_state.processed_data and st.session_state.article_summaries:
    st.subheader("📥 Download Analysis")
    summary_data = []
    for url, info in st.session_state.article_summaries.items():
        summary_data.append({"URL": url, "Title": info['title'], "Summary": info['summary']})
    df = pd.DataFrame(summary_data)
    csv = df.to_csv(index=False)
    st.download_button(label="Download Article Analysis (CSV)", data=csv, file_name="news_analysis.csv", mime="text/csv")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Equity News Research Tool - Powered by Team 5🤖</p>
        <p>Copyright © 2024 Team 5. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
