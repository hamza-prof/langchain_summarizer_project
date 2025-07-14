  name: "🧠 LangChain Text Summarizer"
  description: >
    📝 A modular and customizable text summarization and retrieval project powered by LangChain. 
    Includes an intelligent agent that chooses between LLM-based summarization or 
    RAG (Retrieval-Augmented Generation) depending on user input.

  features:
    - 📄 Load and read plain text documents
    - ✂️ Split long documents into overlapping chunks
    - 🧠 Summarize using customizable LLM prompt templates
    - 🔍 Retrieve document answers using vector search (RAG)
    - 🤖 Intelligent agent chooses between summarizer and retriever
    - ⚙️ Modular architecture for easy extension
    - 🔐 API key and logging support via `.env` and `helpers.py`

  structure:
    root:
      - main.py: 🚀 Entry point script
      - requirements.txt: 📦 Python dependencies
      - .env: 🔐 API keys and secrets
    folders:
      - config:
          - prompts.py: ✏️ Prompt templates and agent system prompt
      - data:
          - sample.txt: 🧪 Example input text
      - modules:
          - loader.py: 📂 Loads and chunks text
          - summarizer.py: 🧠 Summarization logic
          - retriever.py: 🔍 RAG retriever
          - agent.py: 🤖 Agent logic and tool registration
      - outputs:
          - summary.txt: 📝 Summarized output
      - utils:
          - helpers.py: 🔧 Text chunking, logging, env loader

  setup:
    - 🛠️ Clone repository:
        command: git clone https://github.com/your-username/langchain_summarizer_project.git
    - 📁 Navigate into project:
        command: cd langchain_summarizer_project
    - 🧪 Create virtual environment:
        command: python -m venv venv
    - ⚡ Activate environment (Windows):
        command: .\venv\Scripts\activate
    - 📦 Install dependencies:
        command: pip install -r requirements.txt
    - 🔐 Create `.env` file:
        content: |
          OPENAI_API_KEY=your_openai_api_key
          HUGGINGFACEHUB_API_TOKEN=your_huggingface_token

  usage:
    - 🔃 Run full pipeline:
        command: python main.py
    - 🧩 Run loader:
        command: python -m modules.loader
    - 🧠 Run summarizer:
        command: python -m modules.summarizer
    - 🤖 Run agent decision logic:
        command: python -m modules.agent

  output:
    file: outputs/summary.txt
    format: 📋 Bullet-point summary with key ideas preserved

  customization:
    - ✏️ Prompt templates: config/prompts.py
    - ✂️ Chunking strategy: modules/loader.py
    - 🧠 LLM summarizer logic: modules/summarizer.py
    - 🔍 RAG retriever logic: modules/retriever.py
    - 🤖 Agent controller: modules/agent.py

  tools:
    - name: 🧠 summarize_text
      type: LangChain Tool
      source: modules/summarizer.py
      description: Summarizes long text into concise bullet points
    - name: 🔍 retrieve_answer
      type: LangChain Tool
      source: modules/retriever.py
      description: Answers document questions using vector DB (RAG)

  agent:
    file: modules/agent.py
    description: >
      🤖 A LangChain-powered agent that decides whether to summarize or retrieve answers 
      based on the user’s intent. Selects tools using OpenAI Function Calling.
    system_prompt: |
      You are an intelligent assistant.

      If the user's input is long and needs simplification, use the summarizer tool.
      If the user asks a question that might require document retrieval, use the retriever tool.

      Decide wisely. Return only relevant and clear information.
    examples:
      - query: "Please summarize this transcript."
        tool_used: 🧠 summarize_text
      - query: "What did the report say about the budget?"
        tool_used: 🔍 retrieve_answer

  author:
    name: "👨‍💻 Muhammad Hamza"
    github: https://github.com/hamza-prof
    license: MIT
