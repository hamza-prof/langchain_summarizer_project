import os
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_core.runnables import RunnablePassthrough

from modules.RAG.create_pinecone_index import create_pinecone_index
from langchain_core.output_parsers import StrOutputParser

from modules.llm_initializer import LLMInitializer
from modules.loaders import get_text_chunks, load_text_file
from modules.summarizer import combine_summaries, get_summarize_chunks, summarize_documents
from langchain_community.document_loaders import TextLoader


class RAGSystem:
    def __init__(self, k: int = 5, index_name: str = "langchain-summarizer-index", namespace: str = "langchain-summarizer-namespace"):
        self.k=k
        self.retriever = None
        self.rag_chain = None
        self.index_name = index_name
        self.namespace = namespace
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings = HuggingFaceEmbeddings()
        self.llm= LLMInitializer.initialize_llm()
    
    
    def create_vectorstore_from_files(self, file_paths: List[str]) -> PineconeVectorStore:
        print(f"Loading {len(file_paths)} files...")
        
            # Load documents
        docs = load_text_file(file_paths)
            
        # Split documents into chunks
        chunks= get_text_chunks(docs)
            
        # Create or get Pinecone index
        self.index_name = create_pinecone_index(
            self.pc, 
            name=self.index_name, 
            dim=768  # HuggingFace embeddings dimension
        )
            
        # Create vectorstore
        print("Creating vectorstore...")
        self.vectorstore = PineconeVectorStore.from_documents(
            chunks, 
            embedding=self.embeddings, 
            index_name=self.index_name,
            namespace=self.namespace
        )
            
        print("Vectorstore created successfully!")
        return self.vectorstore
    
    def get_retriever_from_existing_index(self, k: int = None) -> Any:

        search_k = k if k is not None else self.k
        
        # Create vectorstore from existing index
        vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": search_k})
        return retriever
    
    def setup_vectorstore_and_retriever(self, file: List[str])-> None:
        self.vectorstore = self.create_vectorstore_from_files(file)
        self.retriever = self.get_retriever_from_existing_index(self.k)
        print("Vectorstore and retriever setup complete.")
    
    
    def setup_from_existing_index(self)-> None:
        
        print(f"Setting up from existing index: {self.index_name}")
        existing_indexes=[index.name for index in existing_indexes]
        if self.index_name not in existing_indexes:
            raise ValueError(f"Index {self.index_name} does not exist in Pinecone. Create it first.")
        self.retriever = self.get_retriever_from_existing_index(k=self.k)
        print("Retriever setup complete from existing index.")
    
    
    def _create_rag_prompt(self) -> ChatPromptTemplate:
        template = """You are an AI assistant that answers questions based on the provided context. 
Use the following context to answer the question. If you cannot find the answer in the context, 
say so clearly and provide any relevant general knowledge if appropriate.

Context:
{context}

Question: {question}

Answer: Provide a comprehensive answer based on the context above. If the context doesn't contain 
enough information to fully answer the question, acknowledge this limitation."""

        return ChatPromptTemplate.from_template(template)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for context."""
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            # Include source information if available
            source = doc.metadata.get('source', 'Unknown source')
            content = doc.page_content.strip()
            formatted_docs.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(formatted_docs) 
    
    def build_rag_chain(self):
        if not self.retriever:
            raise ValueError("Retriever is not set. Call setup_vectorstore_and_retriever() first.")
        
        prompt = self._create_rag_prompt()
        
        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("RAG chain built successfully.")
        
    def query(self, question: str, include_sources: bool = True)-> Dict[str, Any]:
        if not self.rag_chain:
            raise ValueError("RAG chain is not built. Call build_rag_chain() first.")
        
        answer = self.rag_chain.invoke(question)
        result= {
            "answer": answer,
            "question": question,
        }
        
        if include_sources:
            retrieved_docs = self.retriever.invoke(question)
            result["sources"] = [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
            result["num_sources"] = len(retrieved_docs) 
        
        return result


    
    def get_similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform a similarity search on the vectorstore.
        
        Args:
            query: The query string to search for
            k: Number of top results to return
            
        Returns:
            List of Document objects matching the query
        """
        if not self.retriever:
            raise ValueError("Retriever is not set. Call setup_vectorstore_and_retriever() first.")
  
        return self.retriever.invoke(query)
    
    def add_documents_to_existing_vectorstore(self, file_paths: List[str]) -> None:
        """
        Add new documents to an existing vectorstore.
        
        Args:
            file_paths: List of file paths to add
        """
        if not self.vectorstore:
            # Try to connect to existing vectorstore
            try:
                self.vectorstore = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=self.namespace
                )
            except Exception as e:
                raise ValueError(f"No existing vectorstore found. Create one first: {e}")
        
        # Load new documents
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
            print("No new documents were successfully loaded")
            return
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        
        # Add to existing vectorstore
        print(f"Adding {len(chunks)} new chunks to vectorstore...")
        self.vectorstore.add_documents(chunks)
        
        # Reinitialize retriever
        self.retriever = self.get_retriever_from_existing_index(k=self.k)
        
        print("Documents added successfully!")
        
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current Pinecone index.
        
        Returns:
            Dictionary with index statistics
        """
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}
    
    def delete_index(self) -> bool:
        """
        Delete the current Pinecone index.
        
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self.pc.delete_index(self.index_name)
            print(f"Index '{self.index_name}' deleted successfully!")
            # Reset internal state
            self.retriever = None
            self.vectorstore = None
            self.rag_chain = None
            return True
        except Exception as e:
            print(f"Error deleting index: {e}")
            return False
    
    def create_rag_system(self, file_paths: List[str], k: int=5, index_name: str = "langchain-summarizer-index") -> "RAGSystem":
        """
        Create a RAG system with the given file paths.
        
        Args:
            file_paths: List of file paths to create the RAG system from
            k: Number of top results to retrieve
            index_name: Name of the Pinecone index
            namespace: Namespace for the Pinecone index
        """
        rag_system = RAGSystem(k=k, index_name=index_name)
        rag_system.setup_vectorstore_and_retriever(file_paths)
        rag_system.build_rag_chain()
        return rag_system
    
  
# Convenience functions for quick usage
def create_rag_system(file_paths: List[str], k: int = 5, 
                     index_name: str = "langchain-summarizer-index") -> RAGSystem:
    """
    Create and initialize a complete RAG system.
    
    Args:
        file_paths: List of file paths to process
        k: Number of documents to retrieve
        index_name: Name of the Pinecone index
        
    Returns:
        Initialized RAG system
    """
    rag = RAGSystem(k=k, index_name=index_name)
    rag.setup_vectorstore(file_paths)
    rag.build_rag_chain()
    
    return rag


def create_rag_from_existing_index(index_name: str = "langchain-summarizer-index", 
                                  k: int = 5) -> RAGSystem:
    """
    Create RAG system from an existing Pinecone index.
    
    Args:
        index_name: Name of the existing Pinecone index
        k: Number of documents to retrieve
        
    Returns:
        Initialized RAG system
    """
    rag = RAGSystem(k=k, index_name=index_name)
    rag.setup_from_existing_index()
    rag.build_rag_chain()
    return rag


def quick_query(file_paths: List[str], question: str, k: int = 5) -> Dict[str, Any]:
    """
    Quick function to create RAG system and answer a question.
    
    Args:
        file_paths: List of file paths to process
        question: Question to ask
        k: Number of documents to retrieve
        
    Returns:
        Query result
    """
    rag = create_rag_system(file_paths, k)
    return rag.query(question)


# Example usage
if __name__ == "__main__":
    # Example file paths - replace with your actual files
    file_paths = [r"E:\langchain_summarizer_project\data\sample.txt"]
    
    # Method 1: Create RAG system from files
    print("=== Creating RAG system from files ===")
    rag = RAGSystem(k=5)
    rag.setup_vectorstore_and_retriever(file_paths)
    rag.build_rag_chain()
    
    # Method 2: Create RAG system from existing index
    # print("\n=== Creating RAG system from existing index ===")
    # rag_existing = RAGSystem(k=5)
    # try:
    #     rag_existing.setup_from_existing_index()
    #     rag_existing.build_rag_chain()
    #     print("Successfully connected to existing index!")
    # except ValueError as e:
    #     print(f"Could not connect to existing index: {e}")
    
    # Query the system
    question = "What are the main topics discussed in the documents?"
    result = rag.query(question)
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Number of sources used: {result['num_sources']}")
    
    # Example of similarity search
    similar_docs = rag.get_similarity_search("key topic", k=3)
    print(f"\nFound {len(similar_docs)} similar documents")
    
    # Example of document summarization
    summary = summarize_documents(file_paths= file_paths)
    print(f"\nDocument Summary:\n{summary}")
    
    # Get index statistics
    stats = rag.get_index_stats()
    print(f"\nIndex Statistics: {stats}")    