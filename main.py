import os
from modules.loaders import get_text_chunks
from modules.summarizer import combine_summaries, get_summarize_chunks
from utils.helpers import get_logger, load_env


OUTPUT_PATH="outputs/summary.txt"
INPUT_PATH="data/sample.txt"

def main():
    load_env()
    logger= get_logger("LangchainSummarizer")
    logger.info("Starting the Langchain Summarizer application.")
    
    try:
        logger.info(f"📄 Loading document: {INPUT_PATH}")
        chunks = get_text_chunks(INPUT_PATH)
        logger.info(f"✅ Document split into {len(chunks)} chunk(s)")
    except Exception as e:
        logger.error(f"❌ Error loading document: {e}")
        return
    
    try:
        logger.info("🔍 Summarizing chunks...")
        summaries = get_summarize_chunks(chunks)
        logger.info(f"✅ Summarization completed with {len(summaries)} summary(ies)")
    except Exception as e:
        logger.error(f"❌ Error during summarization: {e}")
        
    summary= combine_summaries(summaries)
    
    print (f"📑 Summary:\n{summary}")
    
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"📁 Summary saved to {OUTPUT_PATH}")

    
if __name__ == "__main__":
    main()

