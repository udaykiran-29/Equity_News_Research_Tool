# 📈 Equity News Research Tool

**Equity News Research Tool** is a web-based application built with **Streamlit** that allows users to extract, analyze, and summarize equity-related news articles from URLs or by searching topics. It leverages **LangChain**, **HuggingFace embeddings**, and **FAISS** to build a knowledge base for context-aware question answering.

---

## 🔹 Features

- **Manual URL Input:** Analyze news articles by directly entering URLs.
- **Search By Topic:** Search for relevant articles using NewsAPI.
- **Article Extraction:** Extracts title, summary, authors, and keywords from articles.
- **Contextual Q&A:** Ask questions about the processed articles using language models.
- **Download Analysis:** Export article summaries and analysis results as a CSV.
- **Interactive Interface:** Streamlit-based UI for easy interaction.

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **Streamlit** – Web application framework
- **BeautifulSoup** – Web scraping
- **Pandas** – Data manipulation
- **Requests** – API calls
- **LangChain** – Language model workflows
- **HuggingFace Embeddings** – Vector embeddings for semantic search
- **FAISS** – Fast similarity search for document retrieval
- **Transformers** – Pretrained language models for Q&A
- **dotenv** – Manage API keys and environment variables

---

## 📁 Project Structure


Equity_News_Research_Tool/
├── .env # Environment variables (NewsAPI key)
├── app.py # Main Streamlit application
├──requirements.txt # Python dependencies
├── README.md # Project documentation
├── docs/ # Documentation or additional resources
└── .gitignore # Git ignore file



---

###⚡ Installation

1. Clone the repository:

```bash
git clone https://github.com/udaykiran-29/Equity_News_Research_Tool.git
cd Equity_News_Research_Tool
```

###Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows


### Install dependencies:

pip install -r requirements.txt


### Create a .env file in the root directory with your NewsAPI key And HuggingFace API:

HUGGINGFACEHUB_API_TOKEN=your_api_key_here
NEWS_API_KEY=your_api_key_here

### 🚀 Running the App

streamlit run app.py


## 📁 Project Structure

