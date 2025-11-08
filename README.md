## 🧠 Installation Guide

To set up the environment for this project, install the required dependencies using `pip`.

### 1. Install LangChain and PDF Libraries
```bash
pip install langchain langchain-core langchain-community pypdf pymupdf

| Library                                  | Purpose                                                                                |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| langchain                                | Framework for building LLM-powered applications (RAG, Agents, etc.)                    |
| langchain-core / langchain-community     | Core and community modules providing tools and integrations for LangChain              |
| pypdf / pymupdf                          | For loading and reading PDF files efficiently                                          |
| sentence-transformers                    | Converts text into numerical embeddings (vectors) using pre-trained transformer models |
| faiss-cpu                                | Fast vector similarity search engine (for nearest neighbor retrieval)                  |
| chromadb*                                | Vector database to store and query embeddings efficiently                              |


# Step 1: Clone the repository
git clone https://github.com/RakeshAnvekar/ArtificialIntelligence_Python.git
cd your-repo-name

# Step 2: Install dependencies
pip install -r requirements.txt  # or follow manual installation above

# Step 3: Run your script
python main.py



