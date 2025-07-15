from langchain_core.prompts import PromptTemplate

summarization_prompt = PromptTemplate.from_template(
    """You are an expert text summarizer.

Summarize the following input text while following these rules:
1. Use clear and simple language.
2. Maintain all essential information.
3. Avoid repetition.
4. Structure the summary in bullet points if the content allows.
5. Do not add anything that is not in the original text.

Text:
{text}

Summary:
""".strip()
)