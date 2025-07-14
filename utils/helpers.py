import os
import logging
from dotenv import load_dotenv
from typing import List


def load_env():
    load_dotenv()
    print("Environment variables loaded from .env file.")
    
def get_logger(name: str) -> logging.Logger:
    logger=logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        terminal= logging.StreamHandler()
        terminal.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        terminal.setFormatter(formatter)
        logger.addHandler(terminal)
    
    return logger

def split_text(text: str, chunk_size: int= 1000, chunk_overLap: int = 200):
    chunks=[]
    start=0
    
    while start < len(text):
        end = start +chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start+= chunk_size - chunk_overLap
    return chunks