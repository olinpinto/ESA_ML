import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None
        df = pd.read_parquet(file_path, engine="pyarrow")
        logging.info(f"Loaded data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        return None