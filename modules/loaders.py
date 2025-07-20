import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import helpers
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def load_text_file(file_paths: List[str])-> List[Document]:
        docs = []
        for path in file_paths:
            try:
                loader = TextLoader(path, encoding='utf-8')
                docs.extend(loader.load())
                print(f"Loaded: {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
            
        if not docs:
            raise ValueError("No documents were successfully loaded")
        return docs


def get_text_chunks(docs: Document) ->  List[Document]:
        print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks")
        return chunks

# if (load_text_file("data/sample.txt")):
#     print("Sample text file loaded successfully.")