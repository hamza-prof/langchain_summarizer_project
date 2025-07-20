import os
from langchain_groq import ChatGroq


class LLMInitializer:
    @staticmethod
    def initialize_llm() -> ChatGroq:
        """
        Initializes the Groq LLM with environment variables.
        """
        model = os.getenv("GROQ_MODEL")
        temperature = float(os.getenv("GROQ_TEMPERATURE", 0.3))
        api_key = os.getenv("GROQ_API_KEY")

        if not model or not api_key:
            raise ValueError("GROQ_MODEL and GROQ_API_KEY must be set in environment variables.")

        return ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=temperature,
            max_tokens=512,
        )