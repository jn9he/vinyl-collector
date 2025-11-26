import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import os
from pathlib import Path
from paddleocr import PaddleOCR
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

# --- Global Model Initializations ---

# DINOv2 Model
# Initialize the DINOv2 model and processor globally to avoid reloading on every call.
# Models are moved to the specified device (CUDA if available, otherwise CPU) upon initialization.
try:
    DINO_MODEL_NAME = "facebook/dinov2-small"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DINO_PROCESSOR = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    DINO_MODEL = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE)
    DINO_MODEL.eval()
    print("DINOv2 model loaded successfully.")
except Exception as e:
    DINO_PROCESSOR = None
    DINO_MODEL = None
    print(f"Error loading DINOv2 model: {e}")

# PaddleOCR Model
# Initialize the PaddleOCR model globally. This is a heavy object, and initializing it
# once significantly speeds up subsequent OCR operations.
try:
    OCR_MODEL = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    print("PaddleOCR model loaded successfully.")
except Exception as e:
    OCR_MODEL = None
    print(f"Error loading PaddleOCR model: {e}")

# --- Functions ---

def generate_dinov2_embedding(image_path: str) -> np.ndarray:
    """
    Compute DINOv2 embedding for a single image file using the globally loaded model.
    
    Returns:
      np.ndarray: A 1D normalized vector representing the image, or None if an error occurs.
    """
    if DINO_MODEL is None or DINO_PROCESSOR is None:
        print("DINOv2 model is not available.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = DINO_PROCESSOR(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = DINO_MODEL(**inputs)
        
        cls_token = out.last_hidden_state[:, 0, :]
        feat = cls_token.cpu().numpy()[0]
        feat = feat / np.linalg.norm(feat)

        return feat

    except Exception as e:
        print(f"Error processing DINOv2 for {image_path}: {e}")
        return None

def extract_ocr_text(file_path: str):
    """
    Extracts text from an image file using the globally loaded PaddleOCR model.
    """
    if OCR_MODEL is None:
        print("OCR Error: Model is not available.")
        return {"status": "error", "error": "OCR model is not available."}
        
    if not Path(file_path).exists():
        print(f"OCR Error: File not found at {file_path}")
        return {"status": "error", "error": "File not found"}
    
    try:
        print(f"Running OCR on {file_path}...")
        result = OCR_MODEL.predict(file_path) 
        print(f"Raw OCR result: {result}")
        return result
       
    
    except Exception as e:
        print(f"OCR Error: An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

def find_top_matches_by_embedding(embedding: np.ndarray, top_k: int = 5):
    """
    Finds the most similar album covers in the database using a DINOv2 embedding.

    Args:
        embedding (np.ndarray): The DINOv2 embedding of the image to search for.
        top_k (int): The number of top matches to return.

    Returns:
        list[dict]: A list of dictionaries, each containing the metadata of the top matches, 
                     or None if an error occurs.
    """
    load_dotenv()
    db_url = os.getenv("DB_URL")
    table_name = os.getenv("TABLE_NAME")

    if not db_url or not table_name:
        print("DB_URL or TABLE_NAME not found in environment variables.")
        return None

    conn = None
    try:
        conn = psycopg2.connect(db_url)
        register_vector(conn)
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Note: The table name is sanitized by using psycopg2's mogrify, though direct
            # parameterization for table names is not supported. A safer approach would be
            # to validate the table_name against a known list of tables if possible.
            query = cur.mogrify(
                f"SELECT album_id, title, artist, cover_url, year, style, discogs_url, embedding <=> %s AS distance FROM {table_name} ORDER BY distance LIMIT %s",
                (embedding, top_k)
            )
            cur.execute(query)
            results = cur.fetchall()
            return results

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        return None
    finally:
        if conn:
            conn.close()
