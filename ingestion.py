import os
from dotenv import load_dotenv
load_dotenv()

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore


if __name__ == '__main__':
    print("Ingesting...")

    doc_file_path = os.path.join(os.path.dirname(__file__), "mediumblog1.txt")

    loader = UnstructuredLoader(file_path = doc_file_path, chunking_strategy = "basic", max_characters = 1000000)
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=os.environ.get("GOOGLE_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])

    print ("Finish!")