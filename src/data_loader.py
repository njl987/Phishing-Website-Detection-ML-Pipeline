import pandas as pd
from sqlalchemy import create_engine
from src.config import DB_PATH, SQL_QUERY

def load_data():
    """ Connects to the SQLite database and loads the data into a DataFrame. """
    print(f"Connecting to database at: {DB_PATH}")
    # 'sqlite:///' is the standard format for a file-based SQLite database
    engine = create_engine(f'sqlite:///{DB_PATH}')
    
    try:
        # Note: The raw data comes from a DB, so we use read_sql (as required)
        df = pd.read_sql(SQL_QUERY, engine)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data from database. Ensure data/phishing.db exists. Error: {e}")
        raise
