import os
from pathlib import Path
import pandas as pd
import docx

def task_func(source_directory: str, target_directory: str) -> int:
    if not os.path.exists(source_directory):
        raise FileNotFoundError(f"Source directory '{source_directory}' does not exist.")
    
    Path(target_directory).mkdir(parents=True, exist_ok=True)
    
    file_extensions = ('.txt', '.docx', '.xlsx', '.csv')
    converted_count = 0

    for file in Path(source_directory).glob('*'):
        if file.suffix.lower() in file_extensions:
            target_file = Path(target_directory) / (file.stem + '.csv')
            
            if file.suffix.lower() == '.txt':
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                df = pd.DataFrame([text], columns=['Text'])
                
            elif file.suffix.lower() == '.docx':
                doc = docx.Document(file)
                text = '\n'.join([para.text for para in doc.paragraphs])
                df = pd.DataFrame([text], columns=['Text'])
                
            elif file.suffix.lower() == '.xlsx' or file.suffix.lower() == '.csv':
                df = pd.read_excel(file) if file.suffix.lower() == '.xlsx' else pd.read_csv(file)
                text = df.to_string(index=False)
                df = pd.DataFrame([text], columns=['Text'])
            
            df.to_csv(target_file, index=False)
            converted_count += 1

    return converted_count