# RAG-Pipeline

A basic Retrieval-Augmented Generation (RAG) pipeline for document and web-based knowledge retrieval, chunking, vector embedding, and question answering using Google Generative AI and LangChain.

## Features

- **Document Ingestion**: Load and process local text, PDF, and web documents.
- **Text Chunking**: Split documents into manageable chunks for embedding.
- **Vector Embedding**: Use Google Generative AI Embeddings for semantic search.
- **Vector Stores**: Supports both ChromaDB and FAISS for storing and retrieving document embeddings.
- **Retrieval-Augmented QA**: Retrieve relevant chunks and answer questions using Gemini LLM.
- **Web and PDF Support**: Ingests data from URLs and PDF files.
- **Environment Variable Management**: Uses `.env` for API keys and configuration.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arkapratimdnath/rag-pipeline.git
   cd rag-pipeline
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## Usage

### 1. Data Ingestion
- Load local text files, PDFs, or web pages using LangChain loaders.
- Example (see `ragp.ipynb`):
  ```python
  from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
  loader = TextLoader("Antartica.txt")
  text_document = loader.load()
  ```

### 2. Text Splitting
- Split documents into chunks for embedding:
  ```python
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  documents = text_splitter.split_documents(text_document)
  ```

### 3. Embedding and Vector Store
- Embed and store chunks using Chroma or FAISS:
  ```python
  from langchain_google_genai import GoogleGenerativeAIEmbeddings
  from langchain_community.vectorstores import Chroma, FAISS
  embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
  db = Chroma.from_documents(documents, embeddings)
  # or
  db1 = FAISS.from_documents(documents, embeddings)
  ```

### 4. Retrieval and Question Answering
- Retrieve relevant chunks and answer questions:
  ```python
  from langchain_google_genai import ChatGoogleGenerativeAI
  from langchain_core.prompts import ChatPromptTemplate
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain.chains import create_retrieval_chain

  llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
  prompt = ChatPromptTemplate.from_template("""
  Answer the following question based only on the provided context.\n<context>\n{context}\n</context>\nQuestion: {input}""")
  document_chain = create_stuff_documents_chain(llm, prompt)
  retriever = db.as_retriever()
  retrieval_chain = create_retrieval_chain(retriever, document_chain)
  response = retrieval_chain.invoke({"input": "who are the authors of Attention is all you need paper."})
  print(response["answer"])
  ```

## Example Data
- `Antartica.txt`: Example text file for ingestion.
- `attention.pdf`: Example PDF for ingestion.
- `ragp.ipynb`: Example notebook demonstrating the pipeline.

## Notes
- Make sure your Google API key has access to the required Generative AI models.
- For best results, use a GPU-enabled environment for large documents.

## License
This project is for educational and research purposes.

---

**References:**
- [LangChain Documentation](https://python.langchain.com/)
- [Google Generative AI](https://ai.google.dev/)
- [ChromaDB](https://www.trychroma.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
