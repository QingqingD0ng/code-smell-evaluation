import os
from pathlib import Path
import pandas as pd
from docx import Document

def task_func(source_directory: str, target_directory: str) -> int:
    if not os.path.exists(source_directory):
        raise FileNotFoundError("Source directory does not exist.")
    
    Path(target_directory).mkdir(parents=True, exist_ok=True)
    
    extensions = ['.txt', '.docx', '.xlsx', '.csv']
    converted_count = 0

    for file in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file)
        _, ext = os.path.splitext(file)
        
        if ext.lower() in extensions:
            csv_path = os.path.join(target_directory, f"{Path(file).stem}.csv")
            
            if ext.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                df = pd.DataFrame([text], columns=['Text'])
            
            elif ext.lower() == '.docx':
                doc = Document(file_path)
                text = '\n'.join([para.text for para in doc.paragraphs])
                df = pd.DataFrame([text], columns=['Text'])
            
            elif ext.lower() == '.xlsx':
                df = pd.read_excel(file_path, engine='openpyxl')
                df.to_csv(csv_path, index=False)
                converted_count += 1
                continue
            
            elif ext.lower() == '.csv':
                df = pd.read_csv(file_path)
                df.to_csv(csv_path, index=False)
                converted_count += 1
                continue
            
            df.to_csv(csv_path, index=False)
            converted_count += 1

    return converted_count