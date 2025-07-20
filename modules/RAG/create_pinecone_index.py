from pinecone import Pinecone, ServerlessSpec
import time

def create_pinecone_index(pc: Pinecone, name: str = "langchain-summarizer-index", dim: int = 768) -> str:
    existing = [i["name"] for i in pc.list_indexes()]
    if name not in existing:
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # wait for readiness
        while True:
            status = pc.describe_index(name).status
            if status["ready"]:
                break
            print("Waiting for index to be ready...")
            time.sleep(2)
    return name
