import os
import pandas as pd
import json
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Supabase project URL and anon key
# IMPORTANT: Store these in a .env file in the same directory as this script
# SUPABASE_URL="YOUR_SUPABASE_URL"
# SUPABASE_ANON_KEY="YOUR_SUPABASE_ANON_KEY"
URL: str = os.environ.get("SUPABASE_URL")
KEY: str = os.environ.get("SUPABASE_ANON_KEY")

# Name of the table you want to insert data into
TABLE_NAME = "album_covers"

# Path to your CSV file
CSV_FILE_PATH = "metadata_with_embeddings.csv"

# Column name for the primary key
PRIMARY_KEY = "album_id"

# How many rows to upload in a single batch
BATCH_SIZE = 500

def upload_data():
    """
    Reads data from a CSV file and uploads it in batches to a Supabase table.
    """
    # 1. Initialize Supabase client
    if not URL or not KEY:
        print("Error: SUPABASE_URL and SUPABASE_ANON_KEY must be set in your .env file.")
        return
    
    print("Connecting to Supabase...")
    supabase: Client = create_client(URL, KEY)
    print("Connection successful.")

    # 2. Read and process the CSV file
    print(f"Reading data from {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file {CSV_FILE_PATH} was not found.")
        return

    # --- Data Cleaning and Transformation ---
    # Replace NaN values with None (which will be inserted as NULL in Supabase)
    df = df.where(pd.notna(df), None)

    # Convert the 'embedding' column from a string representation of a list to an actual list
    # and handle any potential errors during the conversion.
    def to_embedding_list(embedding_str):
        try:
            # The string looks like a list, so we can use json.loads
            return json.loads(embedding_str)
        except (TypeError, json.JSONDecodeError):
            # If it's already a list, NaN, or malformed, return None
            return None
            
    print("Transforming embedding data...")
    df['embedding'] = df['embedding'].apply(to_embedding_list)
    
    # Convert 'year' to integer, handling potential None values
    df['year'] = df['year'].apply(lambda x: int(x) if x is not None else None)

    # Filter out rows where the primary key is missing, as they are not valid entries
    original_rows = len(df)
    df.dropna(subset=[PRIMARY_KEY], inplace=True)
    if len(df) < original_rows:
        print(f"Warning: Dropped {original_rows - len(df)} rows due to missing '{PRIMARY_KEY}'.")
        
    # Convert DataFrame to a list of dictionaries for upload
    records = df.to_dict(orient="records")
    total_records = len(records)
    print(f"Found {total_records} records to upload.")

    # 3. Upload data in batches
    for i in range(0, total_records, BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        print(f"Uploading batch {i//BATCH_SIZE + 1}/{(total_records + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} records)...")
        
        try:
            # Use upsert to avoid duplicate entries if the script is run multiple times
            response = supabase.table(TABLE_NAME).upsert(batch, on_conflict=PRIMARY_KEY).execute()
        except Exception as e:
            print(f"An error occurred during batch {i//BATCH_SIZE + 1}: {e}")
            # Optional: Decide if you want to stop or continue on error
            continue

    print("\nUpload complete.")
    print(f"Successfully uploaded {total_records} records to the '{TABLE_NAME}' table.")

if __name__ == "__main__":
    upload_data()
