from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import VARCHAR, INTEGER, TEXT, FLOAT
from sqlalchemy.types import String, Integer, Text
from sqlalchemy.schema import MetaData, Table, Column
from pgvector.sqlalchemy import Vector
import psycopg2
import ast
import pandas as pd
import os
from dotenv import load_dotenv

# DINOv2-small embedding dimension
EMBEDDING_DIM = 384

def push_to_postgres_with_sqlalchemy(df, db_url, table_name):
    engine = create_engine(db_url)

    # First, connect and enable the pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    # Ensure the embedding is of the correct type (list of floats)
    def to_float_list(x):
        if isinstance(x, str):
            x = ast.literal_eval(x)
        return [float(val) for val in x]
    df['embedding'] = df['embedding'].apply(to_float_list)
    
    metadata = MetaData()

    # Define the table with the correct Vector type
    table = Table(
        table_name, metadata,
        Column('album_id', Integer, primary_key=True),
        Column('title', String),
        Column('artist', String),
        Column('cover_url', Text),
        Column('year', Integer),
        Column('style', String),
        Column('discogs_url', Text),
        Column('embedding', Vector(EMBEDDING_DIM)),  # Use Vector type
        extend_existing=True
    )
    
    # Drop the table if it exists and recreate it
    print(f"Dropping and recreating table '{table_name}'...")
    table.drop(engine, checkfirst=True)
    metadata.create_all(engine)
    
    # Use pandas to_sql for efficient insertion
    df.to_sql(
        table_name,
        con=engine,
        if_exists='append',
        index=False,
        method='multi'
    )
    print(f"Data pushed successfully to table '{table_name}'.")


load_dotenv()
final_df = pd.read_csv('metadata_with_embeddings.csv')

DB_URL = os.getenv("DB_URL")
TABLE_NAME = os.getenv("TABLE_NAME")

if not DB_URL or not TABLE_NAME:
    raise EnvironmentError("DB_URL or TABLE_NAME not found in .env file.")

push_to_postgres_with_sqlalchemy(final_df, DB_URL, TABLE_NAME)
