import os
from utils import helpers
from typing import List

def load_text_file(file_path: str)-> str:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
        
    raise FileNotFoundError(f"The file {file_path} does not exist.") 


def get_text_chunks(file_path:str, chunk_size: int = 1000, chunk_overlap: int =200) ->  List[str]:
    text = load_text_file(file_path)
    return helpers.split_text(text, chunk_size, chunk_overlap)  