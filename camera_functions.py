import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import os
from pathlib import Path
from paddleocr import PaddleOCR
from supabase import Client # Correctly import the Supabase client for type hinting
from dotenv import load_dotenv

# --- Global Model Initializations ---

# DINOv2 Model
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
try:
    # Use paddleocr with CPU by default for broader compatibility.
    # use_gpu=False is important if the user doesn't have the CUDA environment set up.
    OCR_MODEL = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    print("PaddleOCR model loaded successfully.")
except Exception as e:
    OCR_MODEL = None
    print(f"Error loading PaddleOCR model: {e}")

# --- Functions ---

def generate_dinov2_embedding(image_path: str) -> np.ndarray:
    """
    Compute DINOv2 embedding for a single image file using the globally loaded model.
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
        # The result from PaddleOCR is a list of lists. We need to process it.
        result = OCR_MODEL.ocr(file_path, cls=True)
        # result is a list of lists, e.g., [[[[points]], ('text', confidence)]]
        # We'll extract the text and confidence.
        if result and result[0]:
            processed_result = [{"rec_texts": [line[1][0] for line in res], "rec_scores": [line[1][1] for line in res]} for res in result]
            print(f"Processed OCR result: {processed_result}")
            return processed_result
        return []
       
    except Exception as e:
        print(f"OCR Error: An exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}

def find_top_matches_by_embedding(supabase_client: Client, embedding: np.ndarray, top_k: int = 5):
    """
    Finds the most similar album covers by calling a Supabase RPC function.

    Args:
        supabase_client (Client): The authenticated Supabase client.
        embedding (np.ndarray): The DINOv2 embedding of the image to search for.
        top_k (int): The number of top matches to return.

    Returns:
        list[dict]: A list of matching items, or None if an error occurs.
    """
    if embedding is None:
        return None
        
    try:
        # Assumes an RPC function `match_metadata` exists in your Supabase project.
        matches = supabase_client.rpc('match_metadata', {
            'query_embedding': embedding.tolist(),
            'match_threshold': 0.7,  # Adjust this threshold as needed
            'match_count': top_k
        }).execute()
        
        print(f"Supabase match results: {matches.data}")
        return matches.data

    except Exception as e:
        print(f"Database error during RPC call: {e}")
        return None

