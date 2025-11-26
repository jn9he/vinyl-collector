import os, time, requests, psycopg2
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import discogs_client
import torch
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv
import re




def clean_styles_data(input_file, output_file='cleaned_styles.csv'):
    """
    Clean the Discogs styles data by extracting just the style names
    
    Args:
        input_file: Path to the input text file with style data
        output_file: Path for the cleaned CSV output
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame with just style names
    """
    styles = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and the header line
            if not line or line.startswith('Style'):
                continue
            
            # Remove all numbers and commas, keep only text
            style_name = re.sub(r'[\d,]+', '', line).strip()
            
            if style_name:
                styles.append(style_name)
    
    # Create DataFrame
    df = pd.DataFrame({'style': styles})

    # Save to CSV
    df.to_csv(output_file, index=False)

    return df

import time
import json

def get_releases_by_style(style, max_results=50):
    releases = []
    # Discogs search usually allows per_page up to 100, but let's stick to your logic
    per_page = min(max_results, 50) 
    
    try:
        results = d.search(style=style, type="release", per_page=per_page)
    except Exception as e:
        print(f"Search failed for style {style}: {e}")
        return []

    yielded = 0
    page = 1
    
    while yielded < max_results:
        # Check pages safely
        if hasattr(results, 'pages') and page > results.pages:
            break

        try:
            print(f"Fetching page {page} for style '{style}'...")
            page_results = results.page(page)
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            time.sleep(5)
            continue

        if not page_results:
            break

        for rel in page_results:
            # 1. Safe Data Access (No API calls)
            # Use .get() on the raw data dictionary to avoid triggering lazy loads
            cover_url = rel.data.get("cover_image") or rel.data.get("thumb")
            year = rel.data.get("year")
            style_data = rel.data.get("style") # Renamed to avoid shadowing 'style' arg
            
            # 2. Artist Extraction Strategy
            artist_name = "Unknown"
            
            # Strategy A: Check raw data (Fastest, no API call)
            if 'artists' in rel.data and len(rel.data['artists']) > 0:
                 artist_name = rel.data['artists'][0].get('name', 'Unknown')
            
            # Strategy B: Parse from Title (Fast, no API call)
            # Search results are usually formatted: "Artist Name - Album Title"
            elif 'title' in rel.data and ' - ' in rel.data['title']:
                try:
                    artist_name = rel.data['title'].split(' - ')[0]
                except:
                    pass

            # Strategy C: The "Heavy" Fetch (Only if absolutely necessary)
            # If we still don't have an artist, we might try the lazy load, 
            # but we MUST sleep to avoid the JSON/RateLimit error.
            if artist_name == "Unknown":
                try:
                    time.sleep(1.5) # Sleep BEFORE the property access triggers the request
                    if rel.artists:
                        artist_name = rel.artists[0].name
                except Exception as e:
                    print(f"Could not fetch artist details: {e}")
                    # Keep 'Unknown' or try to use the raw title as fallback
                    artist_name = rel.data.get('title', 'Unknown')

            data = {
                "album_id": rel.id,
                "title": rel.title,
                "artist": artist_name, # Now passing a String, not a List object
                "cover_url": cover_url,
                "year": year,
                "style": style_data,
                "discogs_url": rel.url,
            }
            
            releases.append(data)
            yielded += 1
            
            if yielded >= max_results:
                break
        
        page += 1
        
        # General page sleep (good to keep even if we sleep inside loop)
        if yielded < max_results:
            time.sleep(2) 

    return releases

def extract_metadata_by_styles(df, max_results=10):
    all_releases = []
    for _, row in df.iterrows():
        style = row['style']
        print(f"Querying style: {style}")
        releases = get_releases_by_style(style, max_results=max_results)
        # Skip releases that don't have a cover URL (cover_url may be None or empty)
        releases = [r for r in releases if r.get('cover_url')]
        all_releases.extend(releases)
        time.sleep(1.2)  # To respect Discogs' API rate limit
    return pd.DataFrame(all_releases)

# Example: styles_df = pd.DataFrame({"style": ["Jazz", "Ambient"]})
# meta_df = extract_metadata_by_styles(styles_df, max_results_per_style=50)
def download_album_covers(metadata_path, output_dir="album_covers"):
    """
    Download album cover images from a metadata file.
    Expects 'album_id' (or 'release_id') and 'cover_url' columns.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    try:
        if metadata_path.endswith('.parquet'):
            metadata_df = pd.read_parquet(metadata_path)
        else:
            metadata_df = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Error loading metadata file: {e}")
        return
    
    # Use a session for connection pooling and optional retries
    session = requests.Session()
    headers = {
        'User-Agent': 'VinylCollector/1.0'
    }

    saved = 0
    skipped = 0
    failed = 0

    print(f"Starting download for {len(metadata_df)} items...")

    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        cover_url = row.get('cover_url')
        
        # Fix: Check for 'album_id' first (which your extraction function uses), 
        # fallback to 'release_id' if needed.
        release_id = row.get('album_id')
        if pd.isna(release_id):
            release_id = row.get('release_id')

        # Normalize and skip missing or invalid IDs/URLs
        if pd.isna(release_id) or not cover_url or str(cover_url).strip().lower() in ['nan', 'none', '']:
            skipped += 1
            continue

        # Ensure ID is a clean string (remove .0 if it was loaded as float)
        if isinstance(release_id, (int, float)):
            release_id = str(int(release_id))
        else:
            release_id = str(release_id).strip()
            
        cover_url = str(cover_url).strip()

        try:
            response = session.get(cover_url, headers=headers, timeout=15, allow_redirects=True)
            if not response.ok or not response.content:
                # print(f"Skipping {cover_url} (status={response.status_code})")
                failed += 1
                continue

            # Determine file extension from content-type header
            content_type = response.headers.get('Content-Type', '').lower()
            if 'png' in content_type:
                ext = '.png'
            elif 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            else:
                # Default to jpg if unknown, or try to guess from URL
                if cover_url.lower().endswith('.png'):
                    ext = '.png'
                else:
                    ext = '.jpg'

            # Construct unique filename using the ID
            fname = os.path.join(output_dir, f"{release_id}{ext}")
            
            # Write file (will create new or overwrite if ID exists)
            with open(fname, 'wb') as f:
                f.write(response.content)
            saved += 1
            
            # Small sleep to be polite to the server
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error downloading {cover_url} for release {release_id}: {e}")
            failed += 1

    print(f"Download complete. Images saved to {output_dir} (saved={saved}, skipped={skipped}, failed={failed})")

# covers_df = download_covers(meta_df, "covers")

def compute_dinov2_embeddings_for_folder(
    metadata_path: str,
    image_dir: str,
    model_name: str = "facebook/dinov2-small",
) -> pd.DataFrame:
    """
    Compute DINOv2 embeddings for images specified in a metadata file and
    return a merged DataFrame with metadata and embeddings.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    metadata_df = pd.read_csv(metadata_path)
    metadata_df['album_id'] = metadata_df['album_id'].astype(str)
    print(f"Loaded {len(metadata_df)} records from {metadata_path}")

    embs = []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    records_to_process = [
        row for _, row in metadata_df.iterrows()
    ]

    for row in tqdm(records_to_process, desc="Computing embeddings"):
        album_id = str(row['album_id'])
        fname = None
        for ext in exts:
            potential_fname = f"{album_id}{ext}"
            path = os.path.join(image_dir, potential_fname)
            if os.path.exists(path):
                fname = potential_fname
                break
        
        if not fname:
            print(f"Warning: No image found for album_id {album_id}. Skipping.")
            embs.append(None)
            continue

        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            feat = out.last_hidden_state[:, 0, :].cpu().numpy()[0]
            feat = feat / np.linalg.norm(feat)
            # Ensure embedding is a list of floats
            embs.append([float(x) for x in feat.tolist()])
        except Exception as e:
            print(f"Error processing {path}: {e}")
            embs.append(None)

    metadata_df['embedding'] = embs
    metadata_df.dropna(subset=['embedding'], inplace=True)
    print(f"Successfully computed embeddings for {len(metadata_df)} records.")

    return metadata_df
def create_discogs_client():
    """
    Create and return a configured discogs_client.Client.

    Looks for credentials in the following environment variables (in order):
      - DISCOGS_USER_TOKEN or DISCOGS_TOKEN : a Discogs personal access token
      - DISCOGS_CONSUMER_KEY and DISCOGS_CONSUMER_SECRET : OAuth consumer creds

    Raises:
        EnvironmentError: if no credentials are found.
    """
    user_token = os.getenv('DISCOGS_USER_TOKEN') or os.getenv('DISCOGS_TOKEN')
    consumer_key = os.getenv('DISCOGS_CONSUMER_KEY')
    consumer_secret = os.getenv('DISCOGS_CONSUMER_SECRET')

    # Provide a descriptive user agent (app name/version and contact URL is recommended)
    user_agent = 'vibeVinyl/1.0 +https://github.com/yourname/vibeVinyl'

    if user_token:
        return discogs_client.Client(user_agent, user_token=user_token)
    elif consumer_key and consumer_secret:
        return discogs_client.Client(user_agent, consumer_key=consumer_key, consumer_secret=consumer_secret)
    else:
        raise EnvironmentError(
            'No Discogs credentials found. Set DISCOGS_USER_TOKEN (or DISCOGS_TOKEN) '
            'or DISCOGS_CONSUMER_KEY and DISCOGS_CONSUMER_SECRET as environment variables.'
        )




if __name__ == "__main__":
    load_dotenv()
    d = create_discogs_client()


    # styles_df = clean_styles_data("/Users/joshnghe/Desktop/Code/personal/portfolio projects/vinyl-detection/.venv/include/vinyl-detection/cleaned_styles.csv")
    # meta_df = extract_metadata_by_styles(styles_df.head(10), max_results=50)

    # meta_df.to_csv('metadata.csv', index=False)
    #download_album_covers('metadata.csv', "/Users/joshnghe/Desktop/Code/personal/portfolio projects/vinyl-detection/covers")

    embeddings_df = compute_dinov2_embeddings_for_folder("metadata.csv", "covers/")
    embeddings_df.to_csv('metadata_with_embeddings.csv', index=False)
    
    # print(embeddings_df.head())
    # print(f"Shape of the final DataFrame: {embeddings_df.shape}")
    # # Verify that the 'embedding' column exists and check one embedding
    # if 'embedding' in embeddings_df.columns and not embeddings_df.empty:
    #     print(f"Sample embedding length: {len(embeddings_df.iloc[0]['embedding'])}")

    # --- Push to PostgreSQL ---
    # IMPORTANT: Replace with your actual database connection details
    # Recommended to use environment variables for this
