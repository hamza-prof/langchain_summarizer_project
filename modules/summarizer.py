import os
from typing import List
from dotenv import load_dotenv
from langchain.chains import LLMChain
from config.prompts import summarization_prompt
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Load env vars once at module load time
load_dotenv()

def load_summarization_chain() -> LLMChain:
    """
    Initializes a summarization chain backed by Groq's LLM.
    """
    model = os.getenv("GROQ_MODEL")
    temperature = float(os.getenv("GROQ_TEMPERATURE"))

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name=model,
        temperature=temperature,
        max_tokens=512,
        reasoning_format="hidden",
    )
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
