import os # used for file and directory operations (like creating folders).
from pathlib import Path # an object-oriented way to handle file paths (better than raw strings).
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader # used to read PDF files and extract text + metadata.
from langchain_text_splitters import RecursiveCharacterTextSplitter #splits long documents into smaller overlapping chunks for RAG
import numpy as np #numerical library used for arrays and mathematical operations
from sentence_transformers import SentenceTransformer #loads a transformer model (like all-MiniLM-L6-v2) for generating embeddings.
import chromadb #a vector database for storing text embeddings and metadata
from chromadb.config import Settings
import uuid #generates unique IDs for each document
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity #measures similarity between embeddings

#########################################-----------------Step 1:  Read All the pdf document inside the folder-------------------#######################################################

def process_all_pdfs(pdf_directory: str): # we pass the path of the folder  like "Data/"
    """Process all the PDF files in a directory."""
    all_documents = [] # place holder to keep all the pdf documents
    pdf_dir = Path(pdf_directory) # Converts your folder path (like "Data/") into a Path object

    # Find all files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf")) #finds all PDFs recursively (even in subfolders)
    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))# is a document loader that helps you read a PDF file and extract its text and metadata into a format LangChain can understand.
            documents = loader.load() #returns a list of Document objects

            # For each document (page), adds extra metadata:
            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents


# Process all PDFs
all_pdf_documents = process_all_pdfs("Data/")

#########################################-----------------Step 2:  Text splitting into chunks-------------------#######################################################


def split_documents(documents, chunk_size=1000, chunk_overlap=200):#Each chunk is around 1000 characters, overlapping by 200.
    """Split documents into smaller chunks for better RAG performance."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] #break text at paragraph, line, or space
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show sample chunk
    if split_docs:
        print(f"\nContent: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs


chunks = split_documents(all_pdf_documents)
print(f"✅ Total chunks created: {len(chunks)}")


# if PDF has 1 page, and that page has 5000 characters

#Chunk  Start index  End index  Characters Covered   Overlap with prev

# 1	    0	        1000	     1000 chars	—
# 2	    800	        1800	     1000 chars	        200 chars overlap
# 3	    1600	    2600	     1000 chars	        200 chars overlap


# why overlap


# Example without overlap:
# Chunk 1: "The company's revenue increased by 20% last year. The main reason"
# Chunk 2: "for this growth was the introduction of a new AI-powered product..."

# When the LLM or vector search looks at one chunk alone, the context is lost — it doesn’t understand the full sentence or meaning.

#With 200-character overlap:
# Chunk 1: "...The company's revenue increased by 20% last year. The main reason"
# Chunk 2: "The main reason for this growth was the introduction of a new AI-powered product..."

# Now both chunks contain the complete context



#########################################-----------------Step 3:  Generating embeddings-------------------#######################################################

# Generating embeddings for text chunks that will later be stored in a vector database (like ChromaDB
# embedding is a numeric representation of text

class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):# this is constructor , will be called once we call te class
        """Initialize the embedding manager."""
        self.model_name = model_name
        self.model = None
        self._load_model() # This line immediately calls the _load_model() method when the object is created..

        #embedding_manager = EmbeddingManager() → This line immediately calls __init__() internally .


    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)# downloads (if not already cached) and loads the model into memory
            print(f"✅ Model loaded! Dimension: {self.model.get_sentence_embedding_dimension()}") #That means each text is converted into a vector of 384 numbers.
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:#This function takes a list of text strings (List[str]) and returns their corresponding embeddings as a NumPy array (np.ndarray)
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Model not loaded")#This ensures the model is actually loaded before trying to use it.If not, it raises an error.

        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)#each text string into a numeric vector
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    #texts = ["I love AI", "I love programming"]
    #“I love AI”                 [0.12, -0.34, 0.88, ..., 0.56]
    #“I enjoy machine learning”  [0.11, -0.32, 0.91, ..., 0.59]


# Initialize the embedding manager
embedding_manager = EmbeddingManager()
print(embedding_manager)

###################################################################################################
# Step 4: Vector Store
class VectorStore:#This class is responsible for storing the document embeddings (from Step 3) into a vector database
    """Manage document embeddings in a ChromaDB vector store."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "Data/vector_store"):
        # pdf_documents is like table name of the database,
        # persist_directory → where the database files are stored on disk
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self): #to actually connect to the ChromaDB instance
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)# Creates the folder (if not already there)
            #Initializes a persistent client that stores data on disk (not just in memory).So your embeddings remain available even after restarting your app.
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            #Either gets an existing collection or creates a new one.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"✅ Vector store initialized | Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    #This is the main function that inserts new data into ChromaDB.
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        #documents: the list of text chunks (each Document from LangChain).
        #embeddings: corresponding numeric vectors for each chunk.
        """
        Add documents and their embeddings to the vector store.
        """
        if len(documents) != len(embeddings):#each document must have one matching embedding
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids, metadatas, documents_text, embeddings_list = [], [], [], []
        # ids: unique document identifiers
        # metadatas: metadata for each document (page, chunk index, etc.)
        # documents_text: actual text content
        # embeddings_list: corresponding embedding vectors

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # documents = ["AI is great.", "Deep learning uses neural nets."]
            # embeddings = [[0.12, -0.35, 0.88], [0.56, -0.21, 0.44]]
            # zip() pairs them together like this:
            #[
                #("AI is great.", [0.12, -0.35, 0.88]),
                #("Deep learning uses neural nets.", [0.56, -0.21, 0.44])
            #]

            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}" #uuid.uuid4() generates a random unique ID (Universally Unique Identifier).just takes the first 8 characters (to keep it short):
            #"5f9c36e377a34d72b08b0f1e4f541f98" => "5f9c36e3"
            # final id is like doc_5f9c36e3_0
            # So every document chunk gets a unique ID like:
                #doc_5f9c36e3_0
                #doc_9a2f11d4_1
                #doc_a8e1b5c7_2
                #This ensures no two documents have the same ID (even across different runs).
            ids.append(doc_id)
            # ids = ["doc_5f9c36e3_0", "doc_9a2f11d4_1", "doc_a8e1b5c7_2"]


            metadata = dict(doc.metadata)# Just ensures we have a clean copy of this metadata that we can modify freely.
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
                # metadatas = [
                #{"source_file": "file1.pdf", "doc_index": 0, "content_length": 900},
                #{"source_file": "file1.pdf", "doc_index": 1, "content_length": 800},
        
                        #]
            # Helps in later filtering, debugging



        try:#This is where ChromaDB actually stores your data permanently.
            # ids ["doc_abc123_0", "doc_def456_1"] Unique ID for each chunk
            # embeddings [[0.12, 0.45], [0.33, 0.22]] Vector representation of each chunk
            # metadatas  [{"source":"file1.pdf"}, {"source":"file2.pdf"}]   Context about each chunk
            # documents ["AI is great", "ML is part of AI"]  Actual text
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"✅ Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in the collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise



# Example usage:
vector_store = VectorStore()
embeddings = embedding_manager.generate_embeddings([chunk.page_content for chunk in chunks])
vector_store.add_documents(chunks, embeddings)