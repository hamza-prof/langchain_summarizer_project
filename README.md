
# 🧠 LangChain Text Summarizer

A modular and customizable text summarization and retrieval project powered by LangChain.  
Includes an intelligent agent that chooses between LLM-based summarization or RAG (Retrieval-Augmented Generation) depending on user input.

---

## 📌 Features

- 📄 Load and read plain text documents
- ✂️ Split long documents into overlapping chunks
- 🧠 Summarize using customizable LLM prompt templates
- 🔍 Retrieve answers from documents using vector search (RAG)
- 🤖 Agent selects either summarizer or retriever tool
- ⚙️ Modular and extensible architecture
- 🔐 API and logging via `.env` and `helpers.py`

---

## 📁 Project Structure

    langchain_summarizer_project/
    ├── main.py               # 🚀 Entry point
    ├── requirements.txt      # 📦 Dependencies
    ├── .env                  # 🔐 API keys and secrets
    ├── config/
    │   └── prompts.py        # ✏️ Prompt templates
    ├── data/
    │   └── sample.txt        # 📄 Example input text
    ├── modules/
    │   ├── loader.py         # 📂 Loads & splits text
    │   ├── summarizer.py     # 🧠 Summarizer tool
    │   ├── retriever.py      # 🔍 RAG retriever tool
    │   └── agent.py          # 🤖 Agent logic and decision-making
    ├── outputs/
    │   └── summary.txt       # 📝 Final output summary
    └── utils/
        └── helpers.py        # 🔧 Logger, chunker, env loader
        ```

---
## ⚙️ Setup

```bash
# 🛠️ Clone the repository
git clone https://github.com/your-username/langchain_summarizer_project.git
cd langchain_summarizer_project

# 🧪 Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # On Windows

# 📦 Install dependencies
pip install -r requirements.txt

# 🔐 Add your API keys to .env
OPENAI_API_KEY=your_openai_key
HUGGINGFACEHUB_API_TOKEN=your_token
````

---

## 🧠 Agent Logic

An intelligent LangChain Agent chooses between:

* 🧠 `summarize_text` → if input is long or structured
* 🔍 `retrieve_answer` → if input is a question

**System prompt** guides tool selection. Example:

```text
User: "Summarize the report"
→ Uses: summarize_text()

User: "What’s mentioned about deadlines?"
→ Uses: retrieve_answer()
```

---

## ✅ Usage

```bash
python main.py              # Run full app
python -m modules.agent     # Use agent logic
python -m modules.loader    # Only load + split
python -m modules.summarizer  # Only summarize
```

---

## 👨‍💻 Author

**Muhammad Hamza**
🔗 [github.com/hamza-prof](https://github.com/hamza-prof)
📜 License: MIT


