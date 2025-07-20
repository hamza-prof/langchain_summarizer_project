import os
from typing import List
from dotenv import load_dotenv
from langchain.chains import LLMChain
from config.prompts import summarization_prompt
from langchain_core.output_parsers import StrOutputParser

from modules.llm_initializer import LLMInitializer
from modules.loaders import get_text_chunks, load_text_file


def load_summarization_chain() -> LLMChain:
    """
    Initializes a summarization chain backed by Groq's LLM.
    """
    llm= LLMInitializer.initialize_llm()
    chain = summarization_prompt | llm | StrOutputParser()
    return chain

def get_summarize_chunks(chunks: List[str]) -> List[str]:
    chain = load_summarization_chain()
    summaries = []
    for idx, chunk in enumerate(chunks, 1):
        try:
            summary = chain.invoke({"text": chunk})
            summaries.append(summary.strip())
        except Exception as e:
            summaries.append(f"[Error summarizing chunk #{idx}: {e}]")
    return summaries

def combine_summaries(summaries: List[str]) -> str:
    return "\n\n".join(f"• {s}" for s in summaries)

def summarize_documents(file_paths: List[str]) -> str:
    # Load documents
    docs = load_text_file(file_paths)
        
    # Split into chunks

    chunks = get_text_chunks(docs)
        
    # Extract text from chunks
        
        
    chunk_texts = [chunk.page_content for chunk in chunks]
        
    # Get summaries
    summaries = get_summarize_chunks(chunk_texts)
        
    # Combine summaries
    combined_summary = combine_summaries(summaries)
        
    return combined_summary


# ---- Hugging Face fallback (commented out) ----
"""
from langchain_huggingface import HuggingFaceEndpoint

def load_summarization_chain_hf() -> LLMChain:
    model_id = os.getenv("HF_MODEL")
    temperature = float(os.getenv("HF_TEMPERATURE", 0.3))
    api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

    llm = HuggingFaceEndpoint(
        repo_id=model_id,
        max_new_tokens=512,
        temperature=temperature,
        huggingfacehub_api_token=api_key
    )
    return summarization_prompt | llm | StrOutputParser()
"""
