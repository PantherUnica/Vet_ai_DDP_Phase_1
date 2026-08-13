"""
Veterinary Inventory OCR and Extraction Pipeline

This script supports multi-model, multi-provider workflows for OCR and inventory extraction.

Model/Provider Selection:
- OCR: Set OCR_PROVIDER and OCR_MODEL (e.g., Mistral, OpenAI Vision, Claude)
- Inventory LLM: Set INVENTORY_LLM_PROVIDER and INVENTORY_LLM_MODEL (e.g., Claude, OpenAI, Mistral)
- Embedding: Set EMBEDDING_PROVIDER and EMBEDDING_MODEL
- Vision fallback: Set VISION_PROVIDER and VISION_MODEL

Performance Optimizations:
- ✅ Parallel OCR processing using ThreadPoolExecutor
- ✅ Parallel Vision model processing using ThreadPoolExecutor (up to 4 workers)
- ✅ Comprehensive timing and logging for performance monitoring
- ✅ Retry logic for API failures and rate limits
- ✅ Error handling with detailed logging

API_Key.txt format (place in the same directory as this script):
MISTRAL_API_KEY=your_mistral_key_here
OPENAI_API_KEY=your_openai_key_here
CLAUDE_API_KEY=your_claude_key_here
(Each key is optional, but required for the provider you select.)

You can mix and match providers/models for each stage of the pipeline by changing the variables below.
"""

# --- Model/Provider Selection ---
OCR_PROVIDER = 'mistral'  # Options: 'mistral', 'openai', 'claude'
OCR_MODEL = 'mistral-ocr-latest'  # e.g., 'mistral-ocr-latest', 'gpt-4o-vision', etc.

INVENTORY_LLM_PROVIDER = 'openai'  # Options: 'mistral', 'openai', 'claude'
INVENTORY_LLM_MODEL = 'gpt-4.1-mini'  # e.g., 'pixtral-12b-latest', 'gpt-4o', 'gpt-5-mini', etc.

EMBEDDING_PROVIDER = 'openai'  # Options: 'openai', ...
EMBEDDING_MODEL = 'text-embedding-3-small'  # Default OpenAI small embedding (1536 dimensions)

VISION_PROVIDER = 'openai'  # Options: 'openai', 'mistral', 'claude'
VISION_MODEL = 'gpt-4.1-nano'  # e.g., 'pixtral-12b-latest', 'gpt-4o-mini', ...

# --- Embedding Activation Switch ---
ENABLE_EMBEDDING = True  # Set to False to disable embedding model or true to enable it

# --- Hybrid Search Configuration ---
# Weight distribution for hybrid search scoring
COSINE_WEIGHT = 0.65      # Weight for cosine similarity (semantic) - increased for OCR robustness
BM25_WEIGHT = 0.35        # Weight for BM25 ranking (keyword) - reduced to balance
BARCODE_BONUS = 0.35      # Bonus added for exact barcode matches

# Enable/disable keyword boosting
ENABLE_KEYWORD_BOOSTING = True  # Set to False to disable boosting entirely

# --- Inventory Management Configuration ---
# Confidence thresholds for different actions
CONFIDENCE_THRESHOLDS = {
    "AUTO_SAVE": 0.5,           # <50% confidence: Save as new product
    "USER_INTERVENTION": 0.8,   # 50-80% confidence: User chooses from top 3
    "AUTO_QUANTITY_ADD": 0.8    # >80% confidence: Auto-add quantity to existing
}

# Batch number field name in the extracted data
BATCH_NUMBER_FIELD = "BatchNumber"  # Adjust based on your LLM extraction
QUANTITY_FIELD = "Quantity"         # Adjust based on your LLM extraction
MRP_FIELD = "MRP"                   # Adjust based on your LLM extraction

# Keyword boosting weights for BM25 search
# These weights are applied during query tokenization (only if ENABLE_KEYWORD_BOOSTING = True)
KEYWORD_BOOSTS = {
    "STN": 4.0,           # STN gets 4x weight (most important)
    "NAME": 3.0,          # Product name gets 3x weight
    "BRAND": 2.0,         # Brand gets 2x weight
    "ACTIVE": 2.0,        # Active ingredients get 2x weight
    "DESC": 1.0           # Description gets 1x weight (baseline)
}

# --- Existing imports and configuration remain unchanged ---
import os
import base64
import requests
import logging
from dotenv import load_dotenv
import openai
import numpy as np
from hybrid_bm25_cosine import HybridBM25Cosine
from bm25_index_manager import bm25_manager
from typing import List, Tuple
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import anthropic
import re
import mimetypes
import difflib
from PIL import Image
import io
from datetime import datetime

# --- Duplicate Detection Integration ---
from consolidated_duplicate_detection import (
    process_duplicate_detection, 
    update_bm25_index_after_product_change,
    save_new_product_to_database,
    add_quantity_to_existing_product,
    add_new_batch_to_existing_product
)

# --- Configuration ---
FOLDER_PATH = r"/Users/vivek/VETINSTANT/wip/New folder/OCR_proto/mistral_ocr_demo/google vision_OCR_test/inventory_test"
MISTRAL_API_URL = "https://api.mistral.ai/v1/ocr"
API_KEY_FILE = os.path.join(os.path.dirname(__file__), 'API_Key.txt')
MERGED_OCR_FILE = "merged_ocr.txt"
FINAL_OUTPUT_FILE = "final_structured_invoice.json"
MAX_VISION_WORKERS = 4  # Maximum number of parallel workers for vision processing

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(FOLDER_PATH, "ocr_processing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Disable logging for specific libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

# --- API Key Loader ---
def load_api_keys(api_key_file: str) -> dict:
    """
    Load API keys from a file.
    Args:
        api_key_file (str): Path to the API key file.
    Returns:
        dict: Dictionary of API keys.
    """
    keys = {}
    if not os.path.exists(api_key_file):
        logger.error(f"API key file not found: {api_key_file}")
        raise FileNotFoundError(f"API key file not found: {api_key_file}")
    with open(api_key_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                keys[k.strip()] = v.strip()
    return keys

# --- Load API keys at startup ---
api_keys = load_api_keys(API_KEY_FILE)
MISTRAL_API_KEY = api_keys.get("MISTRAL_API_KEY", "")
OPENAI_API_KEY = api_keys.get("OPENAI_API_KEY", "")
CLAUDE_API_KEY = api_keys.get("CLAUDE_API_KEY", "")

# --- Utility Functions ---
def encode_file_to_base64(file_path: str) -> str:
    """
    Read a file and return its base64-encoded string.
    Args:
        file_path (str): Path to the file to encode.
    Returns:
        str: Base64-encoded string of the file contents.
    Raises:
        Exception: If file reading or encoding fails.
    """
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode {file_path} to base64: {e}")
        raise

def get_mime_type(file_name: str) -> str:
    """
    Return the MIME type based on file extension.
    Args:
        file_name (str): Name of the file.
    Returns:
        str: MIME type string.
    """
    if file_name.lower().endswith('.png'):
        return "image/png"
    elif file_name.lower().endswith(('.jpg', '.jpeg')):
        return "image/jpeg"
    elif file_name.lower().endswith('.pdf'):
        return "application/pdf"
    else:
        return "application/octet-stream"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(requests.exceptions.RequestException)
)
def process_file_with_ocr(file_path: str, file_name: str) -> str:
    """
    OCR stage: Use the selected provider/model for OCR.
    Extend this function to support other providers/models as needed.
    Args:
        file_path (str): Path to the file to process.
        file_name (str): Name of the file.
    Returns:
        str: OCR-extracted text.
    """
    if OCR_PROVIDER == 'mistral':
        # Use Mistral OCR API
        mime_type = get_mime_type(file_name)
        base64_data = encode_file_to_base64(file_path)
        if mime_type == "application/pdf":
            data = {
                "document": {
                    "type": "document_url",
                    "document_url": f"data:{mime_type};base64,{base64_data}",
                    "document_name": file_name
                },
                "model": OCR_MODEL,
                "include_image_base64": True
            }
        elif mime_type in ("image/jpeg", "image/png"):
            data = {
                "document": {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{base64_data}"
                },
                "model": OCR_MODEL,
                "include_image_base64": True
            }
        else:
            logger.warning(f"Unsupported file type for {file_name}, skipping.")
            return ""
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        response = None
        try:
            response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=120)
            if response.status_code == 429:
                logger.warning(f"429 Too Many Requests for {file_name}. Backing off and retrying...")
                time.sleep(2)
                raise requests.exceptions.RequestException("429 Too Many Requests")
            response.raise_for_status()
            result = response.json()
            logger.info(f"Processed {file_name} with status {response.status_code}")
            ocr_text = []
            if "pages" in result:
                for i, page in enumerate(result["pages"], 1):
                    ocr_text.append(f"--- OCR PAGE {i} TEXT ---\n")
                    ocr_text.append(page.get("markdown", "") or page.get("text", ""))
            else:
                ocr_text.append("No 'pages' key in result!\n")
                ocr_text.append(str(result))
            return "\n".join(ocr_text)
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}\nResponse: {getattr(response, 'text', '') if response else ''}")
            return f"\nError processing {file_name}: {e}\nRaw response text:\n{getattr(response, 'text', '') if response else ''}\n"
    elif OCR_PROVIDER == 'openai':
        logger.error("OpenAI Vision OCR not yet implemented in this script.")
        return ""
    elif OCR_PROVIDER == 'claude':
        logger.error("Claude OCR not yet implemented in this script.")
        return ""
    else:
        logger.error(f"Unknown OCR_PROVIDER: {OCR_PROVIDER}")
        return ""

def process_folder(folder_path: str) -> str:
    """
    Process all supported files in the folder and merge their OCR text.
    Args:
        folder_path (str): Path to the folder containing files.
    Returns:
        str: Merged OCR text from all files.
    """
    merged_ocr_text: List[str] = []
    files_to_process: List[Tuple[str, str]] = []
    for file_name in os.listdir(folder_path):
        lower_name = file_name.lower()
        if lower_name.endswith((".jpg", ".jpeg", ".png", ".pdf")):
            file_path = os.path.join(folder_path, file_name)
            files_to_process.append((file_path, file_name))
        else:
            logger.debug(f"Skipping unsupported file: {file_name}")
    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file_with_ocr, fp, fn): fn for fp, fn in files_to_process}
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                ocr_result = future.result()
                merged_ocr_text.append(ocr_result)
            except Exception as exc:
                logger.error(f"File {file_name} generated an exception: {exc}")
    return "\n".join(merged_ocr_text)

def validate_extracted_data(data: dict) -> dict:
    """
    Validate extracted data for logical consistency and data integrity.
    Args:
        data (dict): Extracted data dictionary.
    Returns:
        dict: Validated data with corrections and warnings.
    """
    validated_data = data.copy()
    warnings = []
    
    # Date validation
    mfg_date = validated_data.get('ManufacturingDate')
    expiry_date = validated_data.get('ExpiryDate')
    
    if mfg_date and expiry_date and mfg_date != 'null' and expiry_date != 'null':
        try:
            # Parse dates (assuming DD-MM-YYYY format)
            mfg_parsed = datetime.strptime(mfg_date, '%d-%m-%Y')
            expiry_parsed = datetime.strptime(expiry_date, '%d-%m-%Y')
            
            if mfg_parsed >= expiry_parsed:
                warnings.append(f"⚠️ Manufacturing date ({mfg_date}) is not before expiry date ({expiry_date})")
                # Try to correct by swapping if they seem reversed
                if mfg_parsed > expiry_parsed:
                    validated_data['ManufacturingDate'] = expiry_date
                    validated_data['ExpiryDate'] = mfg_date
                    warnings.append(f"🔄 Swapped dates: Mfg={expiry_date}, Expiry={mfg_date}")
                    
        except ValueError:
            warnings.append(f"⚠️ Invalid date format: Mfg={mfg_date}, Expiry={expiry_date}")
    
    # Unique identifier validation
    batch_number = validated_data.get('BatchNumber')
    barcode_number = validated_data.get('BarCodeNumber')
    
    if batch_number and barcode_number and batch_number != 'null' and barcode_number != 'null':
        if batch_number == barcode_number:
            warnings.append(f"⚠️ Batch number and barcode number are identical: {batch_number}")
            # Clear one of them (prefer to keep barcode as it's more unique)
            validated_data['BatchNumber'] = None
            warnings.append("🔄 Cleared batch number to maintain uniqueness")
    
    # Additional validations
    unit_price = validated_data.get('UnitSellingPrice')
    if unit_price and unit_price != 'null':
        # Check if price is reasonable (not negative, not too high)
        try:
            price_value = float(unit_price.replace('₹', '').replace(',', ''))
            if price_value < 0:
                warnings.append(f"⚠️ Negative unit price: {unit_price}")
                validated_data['UnitSellingPrice'] = None
            elif price_value > 100000:  # ₹1 lakh seems too high for most vet products
                warnings.append(f"⚠️ Unusually high unit price: {unit_price}")
        except ValueError:
            warnings.append(f"⚠️ Invalid price format: {unit_price}")
    
    # Add validation warnings to the data
    if warnings:
        validated_data['validation_warnings'] = warnings
        logger.info("🔍 Data validation completed with warnings:")
        for warning in warnings:
            logger.info(f"  {warning}")
    else:
        logger.info("✅ Data validation completed - no issues found")
    
    return validated_data

def validate_vision_data(vision_data: dict) -> dict:
    """
    Validate vision-extracted data for consistency.
    Args:
        vision_data (dict): Vision extraction results.
    Returns:
        dict: Validated vision data.
    """
    validated = vision_data.copy()
    
    # Check for duplicate values across different fields
    field_values = {}
    for field, value in validated.items():
        if value and value not in (None, "", "null"):
            if value in field_values:
                logger.warning(f"⚠️ Duplicate value '{value}' found in fields: {field_values[value]} and {field}")
            else:
                field_values[value] = field
    
    return validated

def get_openai_embedding(text: str) -> List[float]:
    """
    Generate OpenAI embedding for text using the configured model and dimensions.
    Args:
        text (str): Input text to embed.
    Returns:
        List[float]: Embedding vector.
    """
    if not text or not isinstance(text, str) or text.strip() == "":
        return []
    
    try:
        # Load OpenAI API key
        api_keys = load_api_keys(API_KEY_FILE)
        openai_api_key = api_keys.get("OPENAI_API_KEY", "")
        
        if not openai_api_key:
            logger.error("OpenAI API key not found")
            return []
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=openai_api_key)
        
        # Clean text
        clean_text = text.replace('\n', ' ').strip()
        
        # Generate embedding with correct dimensions (1536 for text-embedding-3-small)
        response = client.embeddings.create(
            input=clean_text,
            model=EMBEDDING_MODEL,
            dimensions=1536  # Match KB/PAWS pipeline default (full-size small embedding)
        )
        
        embedding = response.data[0].embedding
        logger.info(f"✅ Generated embedding for text (length: {len(clean_text)} chars, dimensions: {len(embedding)})")
        return embedding
        
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {e}")
        return []

def load_existing_products_for_search(limit: int = 1000) -> Tuple[List[str], np.ndarray]:
    """
    Load existing products from database for hybrid search with performance optimization.
    Uses LIMIT to avoid loading all products into memory.
    
    Args:
        limit: Maximum number of products to load (default: 1000)
    
    Returns:
        Tuple of (internaldescriptions, embeddings) for HybridBM25Cosine.
    """
    try:
        import psycopg2
        
        # Database connection (adjust as needed)
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek", 
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            # Get count first
            cursor.execute("""
                SELECT COUNT(*) FROM pm_master 
                WHERE internaldescription IS NOT NULL 
                AND internaldescriptionvector IS NOT NULL
                AND TRIM(internaldescription) != ''
            """)
            total_count = cursor.fetchone()[0]
            
            if total_count == 0:
                logger.warning("⚠️ No existing products found in database for hybrid search")
                return [], np.array([])
            
            # Load limited number of products (most recent first)
            cursor.execute("""
                SELECT internaldescription, internaldescriptionvector
                FROM pm_master 
                WHERE internaldescription IS NOT NULL 
                AND internaldescriptionvector IS NOT NULL
                AND TRIM(internaldescription) != ''
                ORDER BY product_id DESC
                LIMIT %s
            """, (limit,))
            
            results = cursor.fetchall()
            
        conn.close()
        
        internaldescriptions = [row[0] for row in results]
        embeddings = np.array([row[1] for row in results], dtype=np.float32)
        
        logger.info(f"📊 Loaded {len(internaldescriptions)}/{total_count} existing products for hybrid search (limit: {limit})")
        return internaldescriptions, embeddings
        
    except Exception as e:
        logger.error(f"❌ Error loading existing products: {e}")
        return [], np.array([])

def ensure_bm25_index_is_current() -> bool:
    """
    Ensure BM25 index is current and loaded.
    Builds index if needed or if stale.
    
    Returns:
        True if index is ready, False otherwise
    """
    try:
        import psycopg2
        
        # Get current product count
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek", 
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) FROM pm_master 
                WHERE internaldescription IS NOT NULL 
                AND TRIM(internaldescription) != ''
            """)
            current_count = cursor.fetchone()[0]
            
        conn.close()
        
        # Try to load existing index
        if not bm25_manager.load_index():
            logger.info("🔄 Building new BM25 index...")
            return build_bm25_index()
        
        # Check if index is stale
        if bm25_manager.is_index_stale(current_count):
            logger.info(f"🔄 BM25 index is stale (stored: {bm25_manager.metadata.get('total_products', 0)}, current: {current_count}) - rebuilding...")
            return build_bm25_index()
        
        logger.info("✅ BM25 index is current and ready")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error ensuring BM25 index: {e}")
        return False

def build_bm25_index() -> bool:
    """
    Build BM25 index from all products in database.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek", 
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT product_id, internaldescription
                FROM pm_master 
                WHERE internaldescription IS NOT NULL 
                AND TRIM(internaldescription) != ''
                ORDER BY product_id
            """)
            
            results = cursor.fetchall()
            
        conn.close()
        
        if not results:
            logger.warning("No products found for BM25 index")
            return False
            
        product_ids = [row[0] for row in results]
        internaldescriptions = [row[1] for row in results]
        
        # Build index
        success = bm25_manager.build_index(product_ids, internaldescriptions)
        
        if success:
            logger.info(f"✅ BM25 index built successfully for {len(product_ids)} products")
        else:
            logger.error("❌ Failed to build BM25 index")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Error building BM25 index: {e}")
        return False

def perform_efficient_hybrid_search(internaldescription: str, embedding: List[float]) -> dict:
    """
    Perform efficient hybrid search using persistent BM25 index and PostgreSQL vector search.
    This approach is highly scalable and reuses the BM25 index.
    """
    try:
        # Ensure BM25 index is current
        if not ensure_bm25_index_is_current():
            logger.error("❌ Failed to ensure BM25 index is current")
            return {
                "is_duplicate": False,
                "is_new_product": True,
                "top_matches": [],
                "confidence": "unknown",
                "reason": "BM25 index not available"
            }
        
        import psycopg2
        
        # Database connection
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek", 
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            # Convert embedding to PostgreSQL vector format
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'
            
            # Use PostgreSQL's vector similarity search for initial filtering
            cursor.execute("""
                SELECT 
                    product_id,
                    internaldescription, 
                    internaldescriptionvector,
                    1 - (internaldescriptionvector <=> %s::vector) as cosine_similarity
                FROM pm_master 
                WHERE internaldescription IS NOT NULL 
                AND internaldescriptionvector IS NOT NULL
                AND TRIM(internaldescription) != ''
                ORDER BY internaldescriptionvector <=> %s::vector
                LIMIT 100
            """, (embedding_str, embedding_str))
            
            vector_results = cursor.fetchall()
            
        conn.close()
        
        if not vector_results:
            logger.info("🔍 No existing products found - treating as new product")
            return {
                "is_duplicate": False,
                "is_new_product": True,
                "top_matches": [],
                "confidence": "high",
                "reason": "No existing products in database"
            }
        
        # Get BM25 scores using persistent index with optional keyword boosting
        keyword_boosts = KEYWORD_BOOSTS if ENABLE_KEYWORD_BOOSTING else None
        bm25_results = bm25_manager.search(internaldescription, top_k=100, keyword_boosts=keyword_boosts)
        bm25_scores_by_id = {r["product_id"]: r["score"] for r in bm25_results}
        
        # Extract barcode from new product for matching
        new_barcode = _extract_barcode_from_description(internaldescription)
        
        # Process results and combine scores
        results = []
        
        # Normalize BM25 scores to [0,1] range
        bm25_values = list(bm25_scores_by_id.values())
        max_bm25 = max(bm25_values) if bm25_values else 1.0
        min_bm25 = min(bm25_values) if bm25_values else 0.0
        bm25_range = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1.0
        
        for product_id, desc, vec, cosine_sim in vector_results:
            # Get BM25 score for this product
            raw_bm25_score = bm25_scores_by_id.get(product_id, 0.0)
            
            # Normalize BM25 score to [0,1] range
            normalized_bm25 = (raw_bm25_score - min_bm25) / bm25_range if bm25_range > 0 else 0.0
            
            # Check barcode match
            existing_barcode = _extract_barcode_from_description(desc)
            barcode_match = bool(new_barcode and existing_barcode and new_barcode == existing_barcode)
            
            # Combined score using configurable weights (both components are now [0,1])
            combined_score = (COSINE_WEIGHT * cosine_sim) + (BM25_WEIGHT * normalized_bm25)
            if barcode_match:
                combined_score += BARCODE_BONUS
            
            # Ensure score doesn't exceed 1.0
            final_score = min(1.0, combined_score)
            
            results.append({
                "product_id": product_id,
                "score": final_score,
                "barcode_same": barcode_match,
                "bm25_norm": normalized_bm25,
                "bm25_raw": raw_bm25_score,
                "cosine": cosine_sim,
                "candidate_text": desc,
            })
        
        # Sort by score and take top 3
        results.sort(key=lambda x: x["score"], reverse=True)
        top_matches = results[:3]
        
        if not top_matches:
            return {
                "is_duplicate": False,
                "is_new_product": True,
                "top_matches": [],
                "confidence": "high",
                "reason": "No matches found"
            }
        
        # Apply duplicate detection logic
        best_match = top_matches[0]
        best_score = best_match["score"]
        barcode_match = best_match["barcode_same"]
        
        is_duplicate = False
        confidence = "low"
        reason = ""
        
        if barcode_match and best_score > 0.7:
            is_duplicate = True
            confidence = "very_high"
            reason = f"Exact barcode match with high similarity (score: {best_score:.3f})"
        elif barcode_match and best_score > 0.5:
            is_duplicate = True
            confidence = "high"
            reason = f"Exact barcode match with moderate similarity (score: {best_score:.3f})"
        elif best_score > 0.85:
            is_duplicate = True
            confidence = "high"
            reason = f"Very high similarity without barcode match (score: {best_score:.3f})"
        elif best_score > 0.7:
            is_duplicate = True
            confidence = "medium"
            reason = f"High similarity - potential duplicate (score: {best_score:.3f})"
        elif best_score > 0.5:
            confidence = "low"
            reason = f"Moderate similarity - review recommended (score: {best_score:.3f})"
        else:
            confidence = "high"
            reason = f"Low similarity - likely new product (score: {best_score:.3f})"
        
        # Log detailed search results for debugging
        logger.info(f"🔍 HYBRID SEARCH DEBUG:")
        logger.info(f"   Query internaldescription: {internaldescription[:100]}...")
        logger.info(f"   Total products in index: {len(top_matches)}")
        
        if top_matches:
            best_match = top_matches[0]
            logger.info(f"   Best match score: {best_match['score']:.3f}")
            logger.info(f"   Best match barcode: {best_match.get('barcode_same', False)}")
            logger.info(f"   Best match BM25: {best_match.get('bm25_norm', 0):.3f}")
            logger.info(f"   Best match cosine: {best_match.get('cosine', 0):.3f}")
            logger.info(f"   Best match text: {best_match['candidate_text'][:100]}...")
            
            if len(top_matches) > 1:
                logger.info(f"   Second best score: {top_matches[1]['score']:.3f}")
                logger.info(f"   Third best score: {top_matches[2]['score']:.3f}")
        else:
            logger.info(f"   No search results found!")
        
        logger.info(f"🔍 Efficient hybrid search completed: {reason}")
        
        return {
            "is_duplicate": is_duplicate,
            "is_new_product": not is_duplicate,
            "top_matches": top_matches,
            "confidence": confidence,
            "reason": reason,
            "best_score": best_score,
            "barcode_match": barcode_match
        }
        
    except Exception as e:
        logger.error(f"❌ Error in efficient hybrid search: {e}")
        return {
            "is_duplicate": False,
            "is_new_product": True,
            "top_matches": [],
            "confidence": "unknown",
            "reason": f"Search error: {str(e)}"
        }

def _extract_barcode_from_description(description: str) -> str:
    """Extract barcode from internal description (supports 'Barcode: X.' and legacy 'BARCODE: X|')."""
    import re
    # New structured format: "Barcode: 12345." or legacy "BARCODE: 12345|"
    match = re.search(r'BARCODE:\s*([^.|]+)', description or "", re.I)
    if match:
        barcode = match.group(1).strip()
        return re.sub(r'[^0-9a-z]', '', barcode.lower())
    return ""

def _calculate_simple_bm25_score(query: str, document: str) -> float:
    """Calculate a simple BM25-like score for text similarity."""
    if not query or not document:
        return 0.0
    
    # Simple token-based scoring
    query_tokens = set(query.lower().split())
    doc_tokens = set(document.lower().split())
    
    if not query_tokens:
        return 0.0
    
    # Calculate overlap ratio
    overlap = len(query_tokens.intersection(doc_tokens))
    return overlap / len(query_tokens)

def determine_inventory_action(search_results: dict, extracted_data: dict) -> dict:
    """
    Determine the appropriate inventory action based on confidence level and match results.
    
    Args:
        search_results: Results from hybrid search
        extracted_data: Extracted product data from LLM
        
    Returns:
        Dictionary with action type and details
    """
    try:
        confidence = search_results.get("best_score", 0.0)
        is_duplicate = search_results.get("is_duplicate", False)
        top_matches = search_results.get("top_matches", [])
        
        # Extract batch and quantity information
        batch_number = extracted_data.get(BATCH_NUMBER_FIELD, "")
        quantity = extracted_data.get(QUANTITY_FIELD, "")
        mrp = extracted_data.get(MRP_FIELD, "")
        
        logger.info(f"🔍 Analyzing inventory action: confidence={confidence:.3f}, batch={batch_number}, qty={quantity}")
        
        # Decision logic based on confidence levels
        if confidence < CONFIDENCE_THRESHOLDS["AUTO_SAVE"]:
            # <50% confidence: Save as new product
            action = {
                "action_type": "SAVE_NEW_PRODUCT",
                "confidence": confidence,
                "reason": f"Low confidence ({confidence:.1%}) - treating as new product",
                "details": {
                    "batch_number": batch_number,
                    "quantity": quantity,
                    "mrp": mrp,
                    "requires_user_confirmation": False
                }
            }
            logger.info(f"✅ Action: SAVE_NEW_PRODUCT - {action['reason']}")
            
        elif confidence >= CONFIDENCE_THRESHOLDS["AUTO_QUANTITY_ADD"]:
            # >=90% confidence: Auto-add quantity to existing product
            best_match = top_matches[0] if top_matches else None
            if not best_match:
                # Fallback to save new if no matches
                action = {
                    "action_type": "SAVE_NEW_PRODUCT",
                    "confidence": confidence,
                    "reason": "High confidence but no matches found - saving as new",
                    "details": {
                        "batch_number": batch_number,
                        "quantity": quantity,
                        "mrp": mrp,
                        "requires_user_confirmation": False
                    }
                }
            else:
                # Check if batch number matches
                existing_batch = _get_batch_number_from_database(best_match["product_id"])
                batch_match = batch_number and existing_batch and batch_number.lower() == existing_batch.lower()
                
                logger.info(f"🔍 Batch comparison: New='{batch_number}' vs Existing='{existing_batch}' (Match: {batch_match})")
                
                if batch_match:
                    # Same batch: Add quantity to existing
                    action = {
                        "action_type": "ADD_QUANTITY_EXISTING",
                        "confidence": confidence,
                        "reason": f"High confidence ({confidence:.1%}) with matching batch - adding quantity",
                        "details": {
                            "target_product_id": best_match["product_id"],
                            "target_product_text": best_match["candidate_text"],
                            "batch_number": batch_number,
                            "quantity_to_add": quantity,
                            "requires_user_confirmation": False
                        }
                    }
                    logger.info(f"✅ Action: ADD_QUANTITY_EXISTING - Same batch {batch_number}")
                else:
                    # Different batch: Add new batch entry
                    action = {
                        "action_type": "ADD_NEW_BATCH",
                        "confidence": confidence,
                        "reason": f"High confidence ({confidence:.1%}) but different batch - adding new batch entry",
                        "details": {
                            "target_product_id": best_match["product_id"],
                            "target_product_text": best_match["candidate_text"],
                            "new_batch_number": batch_number,
                            "new_quantity": quantity,
                            "new_mrp": mrp,
                            "requires_user_confirmation": False
                        }
                    }
                    logger.info(f"✅ Action: ADD_NEW_BATCH - Different batch {batch_number} vs {existing_batch}")
            
        else:
            # 50-80% confidence: User intervention required
            action = {
                "action_type": "USER_INTERVENTION",
                "confidence": confidence,
                "reason": f"Medium confidence ({confidence:.1%}) - user intervention required",
                "details": {
                    "top_matches": top_matches[:3],  # Show top 3 matches
                    "new_product_data": {
                        "batch_number": batch_number,
                        "quantity": quantity,
                        "mrp": mrp
                    },
                    "requires_user_confirmation": True,
                    "user_options": [
                        "SAVE_NEW_PRODUCT",
                        "ADD_QUANTITY_EXISTING", 
                        "ADD_NEW_BATCH"
                    ]
                }
            }
            logger.info(f"⚠️ Action: USER_INTERVENTION - {action['reason']}")
        
        return action
        
    except Exception as e:
        logger.error(f"❌ Error determining inventory action: {e}")
        return {
            "action_type": "ERROR",
            "confidence": 0.0,
            "reason": f"Error in action determination: {str(e)}",
            "details": {
                "requires_user_confirmation": True
            }
        }

def _extract_batch_from_description(description: str) -> str:
    """Extract batch number from internal description format."""
    import re
    # Try different patterns for batch number extraction
    patterns = [
        r'BATCH:\s*([^|]+)',           # BATCH: pattern
        r'BatchNumber:\s*([^|]+)',     # BatchNumber: pattern  
        r'Batch:\s*([^|]+)',           # Batch: pattern
        r'GW-\d+',                     # Direct GW-XXXX pattern
        r'B\d+',                       # Direct BXXXX pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, description or "", re.I)
        if match:
            return match.group(1).strip() if match.groups() else match.group(0).strip()
    
    return ""

def simulate_user_decision(top_matches: list, product_data: dict) -> dict:
    """
    Simulate user decision for USER_INTERVENTION scenario.
    In a real implementation, this would be replaced with actual user input.
    
    This function implements intelligent decision logic based on:
    1. Match scores and confidence levels
    2. Batch number comparisons
    3. Product similarity analysis
    
    Returns:
        dict: {"action": str, "target_product_id": str, "reason": str}
    """
    try:
        # Get current product data
        current_batch = product_data.get(BATCH_NUMBER_FIELD, "")
        current_quantity = product_data.get(QUANTITY_FIELD, "")
        
        # Analyze top matches
        if not top_matches:
            return {"action": "SAVE_NEW_PRODUCT", "target_product_id": None, "reason": "No matches found"}
        
        best_match = top_matches[0]
        best_score = best_match["score"]
        best_product_id = best_match["product_id"]
        
        # Get batch number from best match
        existing_batch = _get_batch_number_from_database(best_product_id)
        batch_match = current_batch and existing_batch and current_batch.lower() == existing_batch.lower()
        
        logger.info(f"🔍 USER DECISION ANALYSIS:")
        logger.info(f"   Best match score: {best_score:.1%}")
        logger.info(f"   Current batch: '{current_batch}'")
        logger.info(f"   Existing batch: '{existing_batch}'")
        logger.info(f"   Batch match: {batch_match}")
        
        # Decision logic based on score
        if best_score >= 0.5:  # Medium to high confidence match
            # User chooses existing product - system auto-handles batch logic
            return {
                "action": "EXISTING_PRODUCT",
                "target_product_id": best_product_id,
                "reason": f"Confidence ({best_score:.1%}) - add to existing product (auto-handles batch)"
            }
        
        else:  # Low confidence match
            # Low confidence - save as new product
            return {
                "action": "SAVE_NEW_PRODUCT",
                "target_product_id": None,
                "reason": f"Low confidence ({best_score:.1%}) - save as new product"
            }
            
    except Exception as e:
        logger.error(f"❌ Error in user decision simulation: {e}")
        return {
            "action": "SAVE_NEW_PRODUCT",
            "target_product_id": None,
            "reason": f"Error in decision logic - defaulting to new product: {str(e)}"
        }

def get_user_input_interactive(top_matches: list, product_data: dict) -> dict:
    """
    Interactive user input function for real implementation.
    This would replace simulate_user_decision in a production system.
    
    Returns:
        dict: {"action": str, "target_product_id": str, "reason": str}
    """
    print("\n" + "="*60)
    print("🔍 USER INTERVENTION REQUIRED")
    print("="*60)
    
    # Display current product
    print(f"\n📦 Current Product:")
    print(f"   Name: {product_data.get('Name', 'N/A')}")
    print(f"   Brand: {product_data.get('Brand', 'N/A')}")
    print(f"   Batch: {product_data.get(BATCH_NUMBER_FIELD, 'N/A')}")
    print(f"   Quantity: {product_data.get(QUANTITY_FIELD, 'N/A')}")
    
    # Display top matches
    print(f"\n📋 Top {len(top_matches)} Similar Products Found:")
    for i, match in enumerate(top_matches, 1):
        print(f"   {i}. Product ID: {match['product_id']} (Score: {match['score']:.1%})")
        print(f"      Text: {match['candidate_text'][:100]}...")
    
    # Display options
    print(f"\n🎯 Available Options:")
    print(f"   1. SAVE_NEW_PRODUCT - Save as completely new product")
    print(f"   2. EXISTING_PRODUCT - Add to existing product (auto-handles batch logic)")
    
    # Get user input
    while True:
        try:
            choice = input(f"\n👤 Enter your choice (1-2): ").strip()
            
            if choice == "1":
                return {"action": "SAVE_NEW_PRODUCT", "target_product_id": None, "reason": "User chose to save as new product"}
            
            elif choice == "2":
                if not top_matches:
                    print("❌ No existing products to add to")
                    continue
                target_id = top_matches[0]["product_id"]
                return {"action": "EXISTING_PRODUCT", "target_product_id": target_id, "reason": "User chose to add to existing product"}
            
            else:
                print("❌ Invalid choice. Please enter 1 or 2")
                
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return {"action": "SAVE_NEW_PRODUCT", "target_product_id": None, "reason": "User cancelled - defaulting to new product"}
        except Exception as e:
            print(f"❌ Error getting user input: {e}")
            return {"action": "SAVE_NEW_PRODUCT", "target_product_id": None, "reason": f"Input error - defaulting to new product: {str(e)}"}

def _get_batch_number_from_database(product_id: str) -> str:
    """Get batch number directly from database for a product."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek",
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            cursor.execute('SELECT "BatchNumber" FROM pm_master WHERE product_id = %s', (product_id,))
            result = cursor.fetchone()
            batch_number = result[0] if result and result[0] else ""
            
        conn.close()
        return batch_number
        
    except Exception as e:
        logger.error(f"❌ Error getting batch number from database: {e}")
        return ""

def perform_hybrid_search(internaldescription: str, embedding: List[float]) -> dict:
    """
    Perform hybrid BM25+cosine search to find potential duplicates.
    Returns search results with duplicate detection logic.
    """
    try:
        # Load existing products
        existing_descriptions, existing_embeddings = load_existing_products_for_search()
        
        if len(existing_descriptions) == 0:
            logger.info("🔍 No existing products to search against - treating as new product")
            return {
                "is_duplicate": False,
                "is_new_product": True,
                "top_matches": [],
                "confidence": "high",
                "reason": "No existing products in database"
            }
        
        # Initialize hybrid search
        hybrid_search = HybridBM25Cosine(existing_descriptions, existing_embeddings)
        
        # Convert embedding to numpy array
        embedding_array = np.array(embedding, dtype=np.float32)
        
        # Perform search
        top_matches = hybrid_search.search(
            internaldescription, 
            embedding_array, 
            return_k=3
        )
        
        if not top_matches:
            logger.info("🔍 No matches found - treating as new product")
            return {
                "is_duplicate": False,
                "is_new_product": True,
                "top_matches": [],
                "confidence": "high",
                "reason": "No matches found"
            }
        
        # Analyze results for duplicate detection
        best_match = top_matches[0]
        best_score = best_match["score"]
        barcode_match = best_match["barcode_same"]
        
        # Duplicate detection logic
        is_duplicate = False
        confidence = "low"
        reason = ""
        
        if barcode_match and best_score > 0.7:
            is_duplicate = True
            confidence = "very_high"
            reason = f"Exact barcode match with high similarity (score: {best_score:.3f})"
        elif barcode_match and best_score > 0.5:
            is_duplicate = True
            confidence = "high"
            reason = f"Exact barcode match with moderate similarity (score: {best_score:.3f})"
        elif best_score > 0.85:
            is_duplicate = True
            confidence = "high"
            reason = f"Very high similarity without barcode match (score: {best_score:.3f})"
        elif best_score > 0.7:
            is_duplicate = True
            confidence = "medium"
            reason = f"High similarity - potential duplicate (score: {best_score:.3f})"
        elif best_score > 0.5:
            confidence = "low"
            reason = f"Moderate similarity - review recommended (score: {best_score:.3f})"
        else:
            confidence = "high"
            reason = f"Low similarity - likely new product (score: {best_score:.3f})"
        
        logger.info(f"🔍 Hybrid search completed: {reason}")
        
        return {
            "is_duplicate": is_duplicate,
            "is_new_product": not is_duplicate,
            "top_matches": top_matches,
            "confidence": confidence,
            "reason": reason,
            "best_score": best_score,
            "barcode_match": barcode_match
        }
        
    except Exception as e:
        logger.error(f"❌ Error in hybrid search: {e}")
        return {
            "is_duplicate": False,
            "is_new_product": True,
            "top_matches": [],
            "confidence": "unknown",
            "reason": f"Search error: {str(e)}"
        }

def add_quantity_to_existing_product(product_id: str, quantity: str, batch_number: str) -> bool:
    """
    Add quantity to an existing product's batch.
    Args:
        product_id (str): The existing product ID.
        quantity (str): Quantity to add.
        batch_number (str): Batch number.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek",
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            # Update quantity for existing batch
            update_query = """
                UPDATE pm_master 
                SET quantity = COALESCE(quantity, '0')::numeric + %s::numeric
                WHERE product_id = %s AND "BatchNumber" = %s
            """
            
            cursor.execute(update_query, (quantity, product_id, batch_number))
            conn.commit()
            
            logger.info(f"✅ Added {quantity} to existing product {product_id} (batch: {batch_number})")
            
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error adding quantity to existing product: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def add_new_batch_to_existing_product(product_id: str, new_batch_data: dict) -> bool:
    """
    Add a new batch entry to an existing product.
    Args:
        product_id (str): The existing product ID.
        new_batch_data (dict): New batch information.
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek",
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            # Get existing product data
            cursor.execute("SELECT * FROM pm_master WHERE product_id = %s", (product_id,))
            existing_product = cursor.fetchone()
            
            if not existing_product:
                logger.error(f"❌ Product {product_id} not found")
                return False
            
            # Generate new product_id for the new batch
            import time
            import random
            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            new_batch_product_id = f"{timestamp}{random_suffix}"
            
            # Insert new batch as separate product entry
            insert_query = """
                INSERT INTO pm_master (
                    product_id, "Category", "SubCategory", "Name", "TradeName", "Brand", 
                    "Nature", "MajorActiveIngredients", "BarCodeNumber", "ManufacturingDate", 
                    "ExpiryDate", "BatchNumber", "Saleuom", "UnitSellingPrice", 
                    "ConversionFactor", "AdministeredUOM", "AdministeredTotalUnits", 
                    "BriefDescription", internaldescription, internaldescriptionvector
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            # Use existing product data but update batch-specific fields
            values = (
                new_batch_product_id,
                existing_product[1],  # Category
                existing_product[2],  # SubCategory
                existing_product[3],  # Name
                existing_product[4],  # TradeName
                existing_product[5],  # Brand
                existing_product[6],  # Nature
                existing_product[7],  # MajorActiveIngredients
                existing_product[8],  # BarCodeNumber
                new_batch_data.get("ManufacturingDate", existing_product[9]),
                new_batch_data.get("ExpiryDate", existing_product[10]),
                new_batch_data.get("BatchNumber", existing_product[11]),
                existing_product[12], # Saleuom
                new_batch_data.get("UnitSellingPrice", existing_product[13]),
                existing_product[14], # ConversionFactor
                existing_product[15], # AdministeredUOM
                existing_product[16], # AdministeredTotalUnits
                existing_product[17], # BriefDescription
                existing_product[18], # internaldescription
                existing_product[19]  # internaldescriptionvector
            )
            
            cursor.execute(insert_query, values)
            conn.commit()
            
            logger.info(f"✅ Added new batch {new_batch_data.get('BatchNumber')} to product {product_id} with ID: {new_batch_product_id}")
            
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error adding new batch to existing product: {e}")
        if 'conn' in locals():
            conn.close()
        return False

def save_new_product_to_database(product_data: dict) -> str:
    """
    Save a new product to the pm_master database.
    Args:
        product_data (dict): Extracted product data with all fields.
    Returns:
        str: The generated product_id of the saved product.
    """
    try:
        import psycopg2
        # Generate a new product_id (using timestamp + random for uniqueness)
        import time
        import random
        timestamp = int(time.time())
        random_suffix = random.randint(1000, 9999)
        new_product_id = f"{timestamp}{random_suffix}"
        
        # Database connection
        conn = psycopg2.connect(
            dbname="vetinstant",
            user="vivek",
            password="",
            host="127.0.0.1",
            port="5432"
        )
        
        with conn.cursor() as cursor:
            # Insert new product into pm_master
            insert_query = """
                INSERT INTO pm_master (
                    product_id, "Category", "SubCategory", "Name", "TradeName", "Brand", 
                    "Nature", "MajorActiveIngredients", "BarCodeNumber", "ManufacturingDate", 
                    "ExpiryDate", "BatchNumber", "Saleuom", "UnitSellingPrice", 
                    "ConversionFactor", "AdministeredUOM", "AdministeredTotalUnits", 
                    "BriefDescription", internaldescription, internaldescriptionvector
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """
            
            # Prepare the data
            values = (
                new_product_id,
                product_data.get("Category"),
                product_data.get("SubCategory"),
                product_data.get("Name"),
                product_data.get("TradeName"),
                product_data.get("Brand"),
                product_data.get("Nature"),
                product_data.get("MajorActiveIngredients"),
                product_data.get("BarCodeNumber"),
                product_data.get("ManufacturingDate"),
                product_data.get("ExpiryDate"),
                product_data.get("BatchNumber"),
                product_data.get("Saleuom"),
                product_data.get("UnitSellingPrice"),
                product_data.get("ConversionFactor"),
                product_data.get("AdministeredUOM"),
                product_data.get("AdministeredTotalUnits"),
                product_data.get("BriefDescription"),
                product_data.get("InternalDescription"),
                product_data.get("InternalDescriptionVector")
            )
            
            cursor.execute(insert_query, values)
            conn.commit()
            
            logger.info(f"✅ New product saved to database with ID: {new_product_id}")
            logger.info(f"   Product: {product_data.get('Name')} ({product_data.get('Brand')})")
            
            # Update BM25 index after new product is added
            logger.info("🔄 Updating BM25 index after new product addition...")
            try:
                update_success = update_bm25_index_after_product_change()
                if update_success:
                    logger.info("✅ BM25 index updated successfully")
                else:
                    logger.warning("⚠️ BM25 index update failed - will use cache refresh")
            except Exception as e:
                logger.warning(f"⚠️ BM25 index update error: {e} - will use cache refresh")
            
        conn.close()
        return new_product_id
        
    except Exception as e:
        logger.error(f"❌ Error saving new product to database: {e}")
        if 'conn' in locals():
            conn.close()
        raise

def create_internaldescription(data: dict) -> str:
    """
    Create internaldescription using the canonical local inventory template.

    Template (only non-empty parts, in this order):
      Product: [TradeName] [TradeName].
      Generic: [Name] [Name].
      Category: [Category] / [SubCategory].
      Specialty: [Domain].
      Use Case: [BriefDescription].
      Ingredients: [MajorActiveIngredients].

    This aligns the OCR pipeline with local search + embedding behaviour
    described in LOCAL_EMBEDDING_INTERNAL_DESCRIPTION_WORKFLOW.md.
    """

    def _clean(v):
        if v is None or v == "" or (isinstance(v, str) and v.strip() == "") or v == "null":
            return None
        return str(v).strip()

    trade_name = _clean(data.get("TradeName", ""))
    generic_name = _clean(data.get("Name", ""))  # Generic / item name
    category = _clean(data.get("Category", ""))
    subcategory = _clean(data.get("SubCategory", ""))
    domain = _clean(data.get("Domain", ""))  # Domain / specialty, to be populated by LLM
    brief_desc = _clean(data.get("BriefDescription", ""))
    ingredients = _clean(data.get("MajorActiveIngredients", ""))

    # Category or "Category / SubCategory" when both present
    if category and subcategory:
        category_display = f"{category} / {subcategory}"
    else:
        category_display = category or subcategory

    parts = []

    if trade_name:
        parts.append(f"Product: {trade_name} {trade_name}.")

    if generic_name:
        parts.append(f"Generic: {generic_name} {generic_name}.")

    if category_display:
        parts.append(f"Category: {category_display}.")

    if domain:
        parts.append(f"Specialty: {domain}.")

    if brief_desc:
        parts.append(f"Use Case: {brief_desc}.")

    if ingredients:
        parts.append(f"Ingredients: {ingredients}.")

    internaldescription = " ".join(parts) if parts else ""

    logger.info(f"✅ Created internaldescription (local inventory template, length: {len(internaldescription)} chars)")
    return internaldescription

def load_claude_key(filename="Claude key.txt"):
    """
    Load the Anthropic Claude API key from a file.
    Args:
        filename (str): Path to the Claude key file.
    Returns:
        str or None: The API key if found, else None.
    """
    api_key = None
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY="):
                api_key = line.strip().split("=", 1)[1]
    return api_key

ANTHROPIC_API_KEY = load_claude_key()
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((requests.exceptions.RequestException, openai.APIError))
)
def optimize_image_for_vision(image_path: str, max_size: int = 1024) -> str:
    """
    Downscale image to reduce token usage while preserving text readability.
    Args:
        image_path (str): Path to the image file.
        max_size (int): Maximum size for the long edge (default: 1024px).
    Returns:
        str: Base64-encoded optimized image.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new dimensions maintaining aspect ratio
            width, height = img.size
            if max(width, height) <= max_size:
                # Image is already small enough
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='JPEG', quality=85, optimize=True)
                img_buffer.seek(0)
                return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Calculate scaling factor
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to buffer
            img_buffer = io.BytesIO()
            img_resized.save(img_buffer, format='JPEG', quality=85, optimize=True)
            img_buffer.seek(0)
            
            return base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
    except Exception as e:
        logger.error(f"Error optimizing image {image_path}: {e}")
        # Fallback to original image
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

def extract_fields_with_vision_batch(image_paths: list, fields: list) -> dict:
    """
    Process multiple images in a single Vision API call for better efficiency.
    Args:
        image_paths (list): List of image file paths to process.
        fields (list): List of field names to extract.
    Returns:
        dict: Extracted field values from all images.
    """
    extracted = {}
    start_time = time.time()
    
    try:
        prompt = (
            "You are a professional veterinary inventory data extractor. "
            "Extract ONLY the following fields from the product label or packaging images: "
            f"{', '.join(fields)}. "
            "Return a JSON object with only these fields as keys. If a field is not present, return it as null. "
            "Process all provided images and return the best/most accurate value for each field. "
            "Do not include any extra text."
        )
        
        # Prepare content with text prompt and all images
        content = [{"type": "text", "text": prompt}]
        
        # Add all optimized images to the request
        for img_path in image_paths:
            optimized_image_base64 = optimize_image_for_vision(img_path)
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{optimized_image_base64}"}
            })
        
        if VISION_PROVIDER == 'openai':
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional veterinary inventory data extractor."},                                                                                                      
                    {"role": "user", "content": content}
                ],
                max_tokens=1024,  # More tokens for multiple images
                temperature=0
            )
            text = response.choices[0].message.content.strip()
            
        elif VISION_PROVIDER == 'mistral':
            # Mistral API for Pixtral-12B
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Prepare Mistral API request format
            mistral_messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
            
            data = {
                "model": VISION_MODEL,
                "messages": mistral_messages,
                "max_tokens": 1024,
                "temperature": 0
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            
        else:
            raise ValueError(f"Unsupported vision provider: {VISION_PROVIDER}")
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            extracted = json.loads(match.group(0))
        else:
            try:
                extracted = json.loads(text)
            except Exception:
                extracted = {}
        
        duration = time.time() - start_time
        logger.info(f"🔄 Batch vision extraction for {len(image_paths)} images completed in {duration:.2f}s")
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Error in batch vision extraction (took {duration:.2f}s): {e}")
        raise  # Re-raise for retry logic
    
    return extracted

def aggregate_vision_fields_batch(image_files, fields, batch_size=3):
    """
    Process images in batches for optimal Vision API usage.
    Args:
        image_files (list): List of image file paths.
        fields (list): List of field names to extract.
        batch_size (int): Number of images to process in each batch.
    Returns:
        dict: Dictionary of field values.
    """
    vision_results = {field: None for field in fields}
    total_images = len(image_files)
    
    logger.info(f"🔍 Starting batch vision processing for {total_images} images (batch size: {batch_size})...")
    
    # Process images in batches
    for i in range(0, total_images, batch_size):
        batch_images = image_files[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_images + batch_size - 1) // batch_size
        
        logger.info(f"📦 Processing batch {batch_num}/{total_batches} with {len(batch_images)} images...")
        
        try:
            # Process batch of images in single API call
            batch_result = extract_fields_with_vision_batch(batch_images, fields)
            
            # Update vision_results with first non-null values found
            fields_found_this_batch = []
            for field in fields:
                if field in batch_result and batch_result[field] not in (None, "", "null"):
                    if not vision_results[field]:  # Only update if not already found
                        vision_results[field] = batch_result[field]
                        fields_found_this_batch.append(field)
            
            if fields_found_this_batch:
                logger.info(f"✅ Found {len(fields_found_this_batch)} new fields in batch {batch_num}: {fields_found_this_batch}")
            
        except Exception as e:
            logger.error(f"❌ Batch vision extraction failed for batch {batch_num}: {e}")
    
    fields_found = len([v for v in vision_results.values() if v])
    logger.info(f"🎯 Batch vision processing completed: {fields_found}/{len(fields)} fields extracted from {total_images} images")
    
    # Validate vision data before returning
    validated_vision_results = validate_vision_data(vision_results)
    
    return validated_vision_results

# --- Inventory Extraction Prompt Instructions ---
inventory_prompt_instructions = '''
You are a professional veterinary inventory data extractor. You will be given OCR-extracted text from product images, and your task is to extract and structure all relevant inventory fields for each product. 

Instructions:
- Extract all mandatory fields: BarCodeNumber, BatchNumber, ManufacturingDate, ExpiryDate, UnitSellingPrice, AdministeredUOM, AdministeredTotalUnits, and any other required fields as per the schema.
- Use the Vision checklist provided above as a fallback: If a value is not found in the OCR, use the Vision-extracted value.
- Output a single JSON object per product, following the required schema.
- Do not hallucinate or invent values. If a field is not present in either OCR or Vision, return it as null.
- Ensure all date fields are in DD-MM-YYYY format.
- Do not include any extra text, explanations, or comments in the output—only the structured JSON.

You are a pharmacist at a veterinary clinic. You are given data extracted from multiple images of a single inventory item available in the clinic to categorize and digitize. Your goal is to combine all the data and finally output it in the requested output format (mentioned in Phase B), ensuring that all mandatory fields are accurately and completely filled. There are two phases to this prompt: Phase A – Reasoning, Extraction and Computation, Phase B – Final Output. Please follow them carefully and accurately. 


Phase A – Reasoning, Extraction and Computation: 


Mandatory Fields: 


Category, Subcategory, Name, Trade Name, Nature, Bar Code Number, Billing Unit of Measure (UOM), Conversion Factor, Manufacturing Date, Expiry Date, Unit Selling Price (MRP) 


Additional Fields with Importance Levels: 


High Importance: Batch Number,Brief Description of the Product 


Medium Importance: Major Active Ingredients, Brand


Please follow this step-by-step approach to execute the task completely. 


Layer 1: Data Extraction 


Purpose: Extract all available text and data from the OCR output and vision model output. 


Instructions: As the pharmacist at a veterinary clinic, your first task is to meticulously extract all text and numerical data from the provided OCR output of the inventory item. 


This includes: Product labels and names, Ingredients and compositions, Instructions and dosage recommendations, Dates: manufacturing date (Mfg Date), expiry date, batch number, Barcode or unique product number, Price: unit selling price (MRP), and Billing Unit of Measure (UOM). 


Ensure that all visible information is captured accurately for use in subsequent steps. 

Ensure that, the model recognises the stated items, only if it is mentioned against the stated item in the OCR or it can be construed reasonably.

Batch number, will have connonations specifically being mentioned like Batch no, ser No,Lot number or a variation of those words. Consider the output for batch number field in OCR only if, there is an attached word linked it to the batch number.

Layer 2: Product Identification 


Purpose: Assign extracted data to product specific fields. 


Mandatory Fields in this Layer: Category, Subcategory, Name, Trade Name, Nature,Brief Description of the Product 


Additional Fields in this Layer (Medium Importance): Major Active Ingredients, Brand. 


Instructions: 


Category: Based on the product's medical composition and nature, determine the Category. 


Select the most appropriate option from the following list: Medication, Nutrition and Supplements, Flea & Tick Treatment, Vaccines, Deworming, Medical Supplies, Grooming & Hygiene Care, General Consumables, Fluid Therapy, Lab Supplies, Accessories & Toys, Diet, Pet Supplies, Other Parasite Treatment, Surgical Supplies, OTC Products, Mortuary 


The category should strictly reflect the primary use of the product. 


Do not leave this field blank. 


Subcategory: 


Derive the Subcategory based on the products nature and specific usage. Examples include: preventive, anti-fungal, anti-microbial, eye care, skin care, dental care, multi-vitamin, immune support, accessories, toys, etc. Use the products composition and instructions on the label to determine the most appropriate sub-classification. 


Do not leave this field blank. 


Name: Identify the generic Product Name exactly as it appears on the label. 


Do not leave this field blank. 


Trade Name: 
Construct a consistent and uniquely identifiable Standard Trade Name (STN) using the Trade Name as printed on the product label (with minor cleanups), while adding essential structured details to make each SKU uniquely identifiable.
The goal is to allow standardization and duplicate detection without manually reinterpreting product names.The STN is a single-line string that uniquely distinguishes one product SKU from another. It is used to prevent duplicate entries in the inventory system and to enable accurate product matching across batches.
Construction Format
Assemble the STN by filling the slots in this fixed order. Use a single space between each filled slot. If a slot is not applicable, skip it—but never change the slot order.
The output should consist only the final STN string—no labels, explanations, or comments.
STN = [Brand] [Trade name as stated on the label] [Strength] [Pack Size] [Target Segment] [Variant]
Brand - Use the name exactly as on label (without ™/®), example Royal Canin, Clavamox.
Trade Name –  Product's commercial or trade name exactly as on the label. Remove trademark symbols (™/®). Keep spelling as-is after cleanup.
Strength/size - For medicines: mg/mL or mg strength, For diets: weight (kg/g).
and for accessories: size (e.g. "M", "5 m"). Example : 62.5 mg/ml, 4 kg, M, 5 m
Pack Size - Use format: [number] [unit] [type] → e.g., 10 Tab, 120 g, 1 pc, 4 kg Bag
Target Segment if mentioned specifically on the cover - Species, life-stage, weight group, breed - Dog, Cat, Puppy, ≤10 kg, GSD, Boxer. (Don't consider content mentioned in the dosage instructions)
Variant / Flavour / Colour - Flavor, appearance, or variant where relevant. Chicken,Salmon,Red etc.,
Normalization Rules
Remove ™, ®, ℠ symbols from the Trade Name.
Preserve brand capitalization (capitalize only the first letter of Brand).
Preserve trade name as-is after symbol cleanup.
Use single spacing between slots.
No commas, no extra parentheses, except weight bands like ≤10 kg.
Collapse multiple spaces into one, and strip leading/trailing spaces.
If a slot is not applicable, skip it—but never change the slot order.
Do not leave this field blank.

Nature: 
Choose exactly one of the following standardized codes that best describes the physical or functional form of the product.
Return only the short code. Do not invent or describe new types. This is the product form and is different from the packaging form.
Approved Short Codes:
Tab, Cap, Inj, Susp, Sol, Oint, Spray, Paste, Gel, Powder, DrySyrup, EyeDrop, EarDrop, 
Dry, Wet, Biscuit, Chew, Stick, Bone, 
Leash, Collar, Harness, Bed, Bowl, Toy, Carrier, Crate, Mat, Muzzle, 
Shmp, Wipe, Comb, Brush, Lotion,diaper,dispenser,tool,Device,Ramp,cone. 
Syringe, Bandage, Dressing, Catheter, IVSet, Glove, Thermometer, Nebulizer, Scanner, LitterBox
Select the term that best describes the product's form. If it is not clearly mentioned, derive the nature based on the product usage.
Do not leave this field blank.



Major Active Ingredients (Medium Importance): 


List the Major Active Ingredients of the product. Only include the primary active ingredients and not all components. 


Extract from the ingredients list or composition information on the label. 


Use standard chemical names. 


If not explicitly mentioned, note Null or infer based on product type. 


Brand (Medium Importance): 


Identify the Brand or manufacturer of the product. 


Extract from the packaging or label. 


Use the exact brand name as presented. 


If not available, note Null or leave it blank. 


Brief Description of the Product (High Importance): 
Generate a product description that mandatorily includes the brand, trade name, generic name, active ingredients, and purpose and usage. 
Description should follow this standard:STN(Standard Trade name) +Functionality+usage+target audience+purpose
The description should highlight the product's trade name and and function and target audience (e.g., dogs, cats) as well as include the trade name and generic name mandatorily ensuring it is clear and optimized for inventory and search purposes.

Use information from the packaging, label, or product literature. 

Keep it relevant. 

DO not leave this field blank

HSN Code:
Infer from the product's Category, Sub-category, Nature (Product Form) and the product itself, the Indian GST HSN code using the table below.
The table is not exhaustive but should cover majority of the veterinary items. The HSN code finally to be given as an output has to be only one HSN code. 
Ensure that the closest match is selected. 
The HSN code has to be exactly 6 digits. If the product does not fall in the examples given below, then identify the correct 6-digit heading from the official CBIC / HS tariff hierarchy.

For instance, HSN codes can be,
300490 – veterinary medicaments | 300590 – surgical dressings | 330499 – grooming / cosmetic preps
230990 – animal feed supplements | 230910 – retail dog / cat food
901839 – syringes / cannulae / IV sets | 392690 – other plastic articles (e-collar, bowl, dispenser)
420100 – leashes / collars / harness gear

UQC Code: 
Enter the 3 digit appropriate UQC code as per the Indian GST matching with the sales UOM. Give only one UQC code, which should be the closest match to the sales UOM.


Layer 3: Determining the Billing UOM & Pricing (Exception: Strip of Tablets) 


Purpose: The clinic does not dispense any partial units at all. For instance, any containers, jars, bottles, will be dispensed as it is. 


The clinic does not dispense partial units for any of the products (bottles, jars, vials, etc.). 


Exception: If the product is a strip of tablets, partial dispensing is allowed per tablet. 


The label MRP might be for the entire package or sometimes for a subset of the total quantity (e.g., MRP ₹500 for 10 tablets, but the container holds 100 tablets). 


The Purchaseuom may differ from the clinic's chosen default billing UOM, so you must record how the Purchaseuom maps to these dispensing units. 


Mandatory Fields in this Layer: 


Billing UOM 


Conversion Factor (to show how these units relate, and how they compare to the Purchaseuom) 


Unit Selling Price (MRP) (for the Default Billing UOM) 


Instructions: 


Determine If the Product Is a Strip of Tablets 


If Yes (the label or packaging indicates tablets organized in strips): 


You may have two UOMs: 


First UOM = strip 


Second UOM = tablet (because the clinic can break the strip if needed). 


Always consider the second UOM, the tablet as the default billing UOM. 


If No (e.g., a bottle of liquid, a jar of chews, a single‐use vial, dry food packet. Container or bottle containing tablets etc.): 


You have only one UOM: 


Default Billing UOM = 1 bottle, 1 jar, 1 vial, etc. 


Default Billing UOM: 


The whole unit typically dispensed: 


Examples: 1 bottle (ml), 1 vial, 1 jar, 1 packet, etc. 


Do not break it further. 


Default UOM for Tablet Strips 


If the product is in strips of tablets, Default UOM = tablet. 


Conversion Factor 


Show how the Purchaseuom maps to the Default Billing UOM 


Examples: 


Invoice says: 1 box = 5 strips, each strip has 10 tablets: 


fromPurchaseuom_to_defaultBillingUOM: 1 box = 5 strips and 1 strip = 10 tablets. So, the conversion factor is 1 box = 50 tablets. As part of the final output, you will only send the value of the conversion factor and ignore any UOM related texts, in this case: conversion factor = 50. 


If the Purchaseuom matches the default UOM (e.g., both are bottle), your factor would just be 1 bottle = 1 bottle. As part of the final output, you will only send the value of the conversion factor and ignore any UOM related texts, in this case: conversion factor = 1. 


Unit Selling Price (MRP) 


Locate or compute the price for one Default Billing UOM. 


If the label's MRP references a subset of the total quantity (e.g., ₹500 per 10 tablets, but the product actually has 100 tablets total), you must scale it appropriately: 


If your Default Billing UOM is the entire container (100 tablets), multiply the subset MRP to find the total container MRP: 


Sometimes, the label says, for instance, "MRP ₹500 per 10 tablets," but the container actually holds 100 tablets total. To find the entire container's MRP, use: 


MRP  =  (Total Tablets/Subset Size)  ×  MRP (Subset) 


where: 


Total Tablets = total count in the container (e.g., 100), 


Subset Size = the count for which the label provides an MRP (e.g., 10), 


MRP (Subset) = the MRP given for that subset (e.g., ₹500 per 10 tablets). 


If your Default Billing UOM is exactly that subset size (e.g., 10‐tablet strip), then the MRP is directly the subset's price (₹500). 


MRP Selling Price in case of strip of tablets 


Only relevant if the product is a tablet strip alone. 


Compute the per‐tablet cost: 


MRP UOM (Per-Tablet) Selling Price 


Selling Price = Unit Selling Price (per strip) / (number of tablets in one strip) 


Example: 


If Unit Selling Price (per strip of 10) = ₹100, then the per‐tablet price = 100÷10 = ₹10 


Example A: Strip of Tablets 


Label: MRP ₹500 for 10 tablets, package is 1 strip = 10 tablets. 


Invoice: Box of 5 strips. 


Result: 


FirstUOM = strip. 


SecondUOM = tablet. 


DefaultbillingUOM=tablet 


conversionFactor = 


fromPurchaseuom_to_defaultBillingUOM: 1 box = 5 stripsX10 tablets per strip = 50 tablets, i.e., conversion factor = 50 


unitSellingPrice = 500 (₹500 per strip) ÷ 10 = ₹50 per tablet. 


Example B: Bottle of 30 ml 


Label: MRP ₹300 for 30 ml. 


Invoice: Qty 2 bottles. 


Result: 


defaultBillingUOM = bottle 


conversionFactor = 1 bottle = 1 bottle, i.e., conversion factor = 1 


Example C: 100 Tablets, but MRP is ₹500 per 10 


Label: Box Contains 100 tablets, MRP ₹500 per 10 tablets. 


Clinic Policy: Sells only 1 box containing 100 tablets (no partial usage unless these are physically in strips). 


The key is: Check the actual packaging. 


If the Default Billing UOM is the entire container (100 tablets): 


MRP = (100 ÷ 10) × ₹500 = ₹5,000 per container. 

Layer 3A: Administered UOM & Total Units
Purpose:
Enable veterinarians to administer doses from the product by identifying the smallest measurable or applicable unit and its total quantity per SKU (e.g., mL, mg, tablet, dose, vial, etc.).

Mandatory Fields in this Layer:
AdministeredUOM
AdministeredTotalUnits

Instructions:
AdministeredUOM:
Identify from the product's pack size or strength.
Choose the most clinically administered unit from the following list:
ml, L, g, kg, drops, tablet, capsule, chew/treat, vial, ampoule, pre-filled syringe, tube, sachet, bag, pouch, pipette, patch, spray, scoop, cup, dose, syringe, injection.
If the product says "50 mg/25 mL", the AdministeredUOM is mL (not vial).
Ensure this unit is based on what a vet would physically draw or dose per use.
Do not leave this field blank.

AdministeredTotalUnits:
Derive from the pack size or label, representing the total measurable amount of that UOM in one product unit.
Examples:
If it says 25 mL per vial, then: AdministeredUOM = mL, AdministeredTotalUnits = 25
If it's 10 tablets in 1 strip: AdministeredUOM = tablet, AdministeredTotalUnits = 10

Layer 4: Regulatory and Date Information 


Purpose: Extract manufacturing and expiry dates. 


Mandatory Fields in this Layer: Manufacturing Date, Expiry Date, Bar Code Number 


Additional Field in this Layer (High Importance): Batch Number  


Manufacturing Date: Find the Manufacturing Date (Mfg Date) on the products packaging or label. 


Instructions: 


Extract the Manufacturing Date as provided. Verify if it is the manufacturing date or the expiry date. Do not misread the information presented. 


If the Manufacturing Date is explicitly mentioned on the label, extract it directly. 


If only the month and year are provided, assume the day as 01 (Example: March 2023 becomes 01-03-2023). 


Use the format DD-MM-YYYY for the Manufacturing Date. 


Do not leave this field blank. 


Expiry Date: Determine the Expiry Date for the product. Verify if it is the manufacturing date or the expiry date. Do not misread the information presented on the label. 


Instructions: 


If the Expiry Date is explicitly mentioned, extract it directly. 


If the expiry date is not directly provided but a shelf life is indicated (Example: 12 months from date of manufacturing (mfg date)), then compute the Expiry Date by adding the shelf life to the Manufacturing Date. 


Please note that the expiry date is only a forward calculation from the manufacturing date and not a backward calculation. Please refer to the example given below and ensure that the computed date matches with the given criteria. 


Use the format DD-MM-YYYY for Expiry Date. 

If only the month and year are provided, assume the day as 01 (Example: March 2023 becomes 01-03-2023).

Example: 


If the Manufacturing Date is 15-03-2023 and the shelf life is 12 months, then the Expiry Date is 15-03-2024. 


Ensure the computed Expiry Date is accurate. 


Do not leave this field blank. 


Bar Code Number: Find the Bar Code Number on the products packaging or label. 


Instructions: 


Extract the bar code number or unique product id or number as provided. 


Ensure it is recorded accurately. 


Do not leave this field blank. 


Batch Number (High Importance): Find the Batch Number on the products packaging or label. 


Instructions: 


Extract the batch number as provided. 


Ensure it is recorded accurately. 


If not available, note Null. 


Layer 5: Data Verification and Computation 


Purpose: Verify all data for accuracy and completeness; compute derived values. 


Instructions: 


Review all the information gathered to ensure the following: 


Completeness: All mandatory fields and high importance fields are filled as per the instructions. 


Accuracy: Data accurately reflects the information extracted from the images. 


Consistency: Related fields are logically consistent (e.g., Measurement Units align with the Formula; dates are correctly formatted). 


Calculations: Verify all computations, especially for Subunit Selling Price and Expiry Date, following the provided instructions. 


Formatting: Dates must be in DD-MM-YYYY format. 


Layer 6: Data Formatting and Output 


Purpose: Organize all verified data into the final formatted output before structuring it in the required output format. 


Instructions: This step is to ensure that all the details are extracted, computed and verified before presenting the final output. 


Ensure to follow the naming convention for each data point as provided and combine it as a single word in titlecase. Follow the output structure for all the data points. 


Category: Category content here 


SubCategory: Subcategory content here 


Name: Name content here 


TradeName: Trade Name content here 


Brand: Brand content here 


Nature: Nature content here 


MajorActiveIngredients: Major Active Ingredients content here 


BarCodeNumber: Bar Code Number content here 


ManufacturingDate: Manufacturing Date content here 


ExpiryDate: Expiry Date content here 


BatchNumber: Batch Number content here 


Saleuom: Billing Unit of Measurement (UOM) content here 


UnitSellingPrice: [Unit Selling Price (MRP) content here 


ConversionFactor: Conversion Factor value here 


BriefDescription: Brief Description of the Product content here 


Phase B – Final Output: 

Purpose: Present the extracted data as an array of objects in JSON format mentioned below. Ensure that the given field names are followed accurately for each field and do not add any additional characters or symbols. Ensure to strictly follow the name and letter casing format provided. 

Instructions: 

Strictly include only value for each of the output fields and not any other additional texts. Do not omit any of the fields, strictly follow the output structure. Leave a field blank or return Null if it is unavailable, do not exclude the field or include any other texts. 

Prepare the output by populating values for each of the items against the given fields. Use the exact field names as provided, do not change them. 

Generate the output as per the given schema: 

Example final output format: 

Example final output format:

{{
  "Category": "Vaccines",
  "SubCategory": "Preventive",
  "Name": "CDV-CAV2-CPIV-CPV Vaccine MLV",
  "TradeName": "Zoetis Vanguard DAPP-L4 Injectable",
  "Brand": "Zoetis",
  "Nature": "Injectable",
  "MajorActiveIngredients": "Lcan, Lgrip, Lict, Lpom Bacterin",
  "BarCodeNumber": "40041437",
  "ManufacturingDate": "18-02-2024",
  "ExpiryDate": "12-08-2025",
  "BatchNumber": "7501B23",
  "Saleuom": "Vial",
  "UnitSellingPrice": "₹900",
  "ConversionFactor": "1",
 "AdministeredUOM": "mL",
  "AdministeredTotalUnits": "25",
  "BriefDescription": "A vaccine for dogs to prevent CDV, CAV2, CPIV, and CPV infections commonly referred to as Zoetis Vanguard DAPP-L4 Injectable."
}}


Mandatory Note: Provide only the output information and do not provide any additional conversational texts at the beginning or end of the output. Strictly provide only the output information as per the given example structure as an object. Do not include any other additional texts or fields in the final output, strictly provide only the specified fields and their extracted or computed values. Do not miss any of the fields, return it as Null or leave it blank, but do not omit the field. 

Final Reminders: 


Strictly do not include any additional texts or reasoning to the final output. The final output should only be the JSON format of the extracted information. 


Accuracy is Essential. 


Ensure all information is accurate and consistent. 


Ensure the following: 


Clarity: The table is clear and easy to read. 


Alignment: All data is correctly aligned under the appropriate field names. 


Completeness: Ensure all required fields are filled. If its unavailable, leave it blank or return Null, but do provide a random output. 


Professionalism: The final output is professional and suitable for inclusion in the clinics inventory system. Use correct medical and pharmaceutical terminology throughout. 


Final Notes: 


Ensure Compliance with Instructions: Follow the detailed instructions for all fields, prioritizing mandatory and high-importance fields. 


Consistency in Units and Terms: Use standard units and terms as specified. 


Accuracy in Calculations and Dates: Double check all computations and date formats. 


Handling Missing Information: If certain information is missing and cannot be found or inferred, clearly indicate Null to maintain data integrity. 


Professional Presentation: The final table should be suitable for official records and easy to reference and should be strictly as per the given output format.   


If any mandatory fields are not filled, reread the extracted text again to identify the correct value. If it is still unavailable, then leave it blank or return Null, but do not include random values.
'''


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(10),
    retry=retry_if_exception_type((anthropic.APIStatusError, anthropic.RateLimitError, openai.APIError, openai.RateLimitError, requests.exceptions.RequestException))
)
def send_to_llm(merged_ocr_text: str, folder_path: str, vision_fields: dict = None) -> str:
    """
    Send the merged OCR text to the configured LLM provider for inventory extraction and save the result.
    Uses intelligent source selection between OCR and Vision data.
    Dynamically switches between Claude and OpenAI based on INVENTORY_LLM_PROVIDER configuration.
    Args:
        merged_ocr_text (str): The merged OCR text from all files.
        folder_path (str): The folder path to save outputs.
        vision_fields (dict): Vision-extracted structured data for intelligent source selection.
    Returns:
        str: The raw LLM output (JSON as string or fallback text).
    """
    # Create intelligent source selection prompt
    if vision_fields:
        intelligent_prompt = f"""
You have TWO data sources for the same product:

## SOURCE A - RAW OCR TEXT:
{merged_ocr_text}

## SOURCE B - VISION-EXTRACTED STRUCTURED DATA:
{json.dumps(vision_fields, indent=2)}

## INTELLIGENT SOURCE SELECTION TASK:
For each field you extract, intelligently choose the BEST source:

**Generally prefer OCR for:** Numbers, prices, batch codes, barcodes, dates
**Generally prefer Vision for:** Product names, brands, categories (less spelling errors)

**Decision rules:**
- If both sources have the same field, pick the one that looks more accurate
- If only one source has a field, use that source
- For critical numeric data (MRP, batch codes, dates), be extra careful about accuracy
- If unsure between sources, mention both values and mark for verification

**CRITICAL VALIDATION REQUIREMENTS:**
- **Date Validation:** Manufacturing date MUST be before expiry date. If dates appear reversed, swap them.
- **Unique Identifiers:** Batch number and barcode number CANNOT be identical. If they are the same, prefer barcode and clear batch number.
- **Price Validation:** Unit selling price must be positive and reasonable (not negative, not excessively high).
- **Cross-field Validation:** Ensure no duplicate values across different fields (e.g., same value for Name and Brand).

**Data Integrity Checks:**
- Verify all dates are in DD-MM-YYYY format
- Ensure batch numbers are alphanumeric codes (not just numbers)
- Validate that barcode numbers are typically longer numeric sequences
- Check that product names and trade names are different (not identical)

{inventory_prompt_instructions}
"""
    else:
        # Fallback to original approach if no vision fields provided
        intelligent_prompt = inventory_prompt_instructions + merged_ocr_text
    
    try:
        logger.info(f"🤖 Sending request to {INVENTORY_LLM_PROVIDER.upper()} LLM...")
        
        if INVENTORY_LLM_PROVIDER == 'claude':
            # Initialize Claude client
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            
            response = client.messages.create(
                model=INVENTORY_LLM_MODEL,
                max_tokens=4096,
                temperature=0,
                system="You are a professional data extraction specialist with expertise in choosing optimal data sources.",
                messages=[
                    {"role": "user", "content": intelligent_prompt}
                ]
            )
            final_output = response.content[0].text.strip()
            
        elif INVENTORY_LLM_PROVIDER == 'openai':
            # Initialize OpenAI client
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.chat.completions.create(
                model=INVENTORY_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional data extraction specialist with expertise in choosing optimal data sources."},
                    {"role": "user", "content": intelligent_prompt}
                ],
                max_tokens=4096,
                temperature=0
            )
            final_output = response.choices[0].message.content.strip()
            
        elif INVENTORY_LLM_PROVIDER == 'mistral':
            # Mistral API for Pixtral-12B
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Prepare Mistral API request format
            mistral_messages = [
                {
                    "role": "user",
                    "content": intelligent_prompt
                }
            ]
            
            data = {
                "model": INVENTORY_LLM_MODEL,
                "messages": mistral_messages,
                "max_tokens": 4096,
                "temperature": 0
            }
            
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            final_output = result["choices"][0]["message"]["content"].strip()
            
        else:
            raise ValueError(f"Unsupported LLM provider: {INVENTORY_LLM_PROVIDER}")
        
        logger.info(f"✅ {INVENTORY_LLM_PROVIDER.upper()} LLM response received successfully")
        # Save outputs
        merged_path = os.path.join(folder_path, MERGED_OCR_FILE)
        with open(merged_path, "w", encoding="utf-8") as f:
            f.write(merged_ocr_text)
        final_path = os.path.join(folder_path, FINAL_OUTPUT_FILE)
        # Try to parse the output as JSON and add the embedding
        try:
            data = json.loads(final_output)
            
            # Validate the extracted data
            logger.info("🔍 Validating extracted data for consistency...")
            validated_data = validate_extracted_data(data)
            
            # Create internaldescription field
            logger.info("🔗 Creating internaldescription field...")
            internaldescription = create_internaldescription(validated_data)
            validated_data["InternalDescription"] = internaldescription
            
            # Generate embeddings if enabled
            if ENABLE_EMBEDDING and internaldescription:
                logger.info("🧠 Generating embeddings for internaldescription...")
                validated_data["InternalDescriptionVector"] = get_openai_embedding(internaldescription)
                
                # Use the new separate duplicate detection module
                logger.info("🔍 Processing duplicate detection with separate module...")
                try:
                    duplicate_result = process_duplicate_detection(validated_data, OPENAI_API_KEY)
                    
                    # Add duplicate detection results
                    validated_data["DuplicateDetection"] = {
                        "is_duplicate": duplicate_result["is_duplicate"],
                        "confidence": duplicate_result["confidence"],
                        "action_type": duplicate_result["action_type"],
                        "action_reason": duplicate_result["action_reason"],
                        "requires_user_intervention": duplicate_result["requires_user_intervention"],
                        "product_id": duplicate_result["product_id"],
                        "top_matches": duplicate_result["top_matches"],
                        "best_match": duplicate_result["best_match"]
                    }
                    
                    # Add inventory action
                    validated_data["InventoryAction"] = {
                        "action_type": duplicate_result["action_type"],
                        "action_reason": duplicate_result["action_reason"],
                        "requires_user_intervention": duplicate_result["requires_user_intervention"],
                        "product_id": duplicate_result["product_id"],
                        "batch_number": duplicate_result["batch_number"],
                        "quantity": duplicate_result["quantity"]
                    }
                    
                    # Log the results
                    if duplicate_result["is_duplicate"]:
                        logger.warning(f"⚠️ POTENTIAL DUPLICATE DETECTED: {duplicate_result['action_reason']}")
                        if duplicate_result["top_matches"]:
                            best_match = duplicate_result["top_matches"][0]
                            logger.warning(f"   Best match: Score {best_match['score']:.3f}, Barcode match: {best_match['barcode_same']}")
                    else:
                        logger.info(f"✅ NEW PRODUCT: {duplicate_result['action_reason']}")
                    
                    # Log inventory action
                    action_type = duplicate_result["action_type"]
                    confidence = duplicate_result["confidence"]
                    logger.info(f"🎯 INVENTORY ACTION: {action_type} (confidence: {confidence:.1%})")
                    
                    if duplicate_result["requires_user_intervention"]:
                        logger.warning(f"👤 USER INTERVENTION REQUIRED: {duplicate_result['action_reason']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error in duplicate detection module: {e}")
                    # Fallback to old system if new module fails
                    logger.info("🔄 Falling back to integrated duplicate detection...")
                    search_results = perform_efficient_hybrid_search(internaldescription, validated_data["InternalDescriptionVector"])
                    validated_data["DuplicateDetection"] = search_results
                    inventory_action = determine_inventory_action(search_results, validated_data)
                    validated_data["InventoryAction"] = inventory_action
                    
                    # Log the results for fallback
                    if search_results["is_duplicate"]:
                        logger.warning(f"⚠️ POTENTIAL DUPLICATE DETECTED: {search_results['reason']}")
                        if search_results["top_matches"]:
                            best_match = search_results["top_matches"][0]
                            logger.warning(f"   Best match: Score {best_match['score']:.3f}, Barcode match: {best_match['barcode_same']}")
                    else:
                        logger.info(f"✅ NEW PRODUCT: {search_results['reason']}")
                    
                    # Log inventory action
                    action_type = inventory_action["action_type"]
                    confidence = inventory_action["confidence"]
                    logger.info(f"🎯 INVENTORY ACTION: {action_type} (confidence: {confidence:.1%})")
                    
                    if action_type == "USER_INTERVENTION":
                        logger.warning(f"👤 USER INTERVENTION REQUIRED: {inventory_action['reason']}")
                    
                    # Log the options for user review
                    top_matches = inventory_action["details"]["top_matches"]
                    logger.info(f"📋 Top {len(top_matches)} matches found:")
                    for i, match in enumerate(top_matches, 1):
                        logger.info(f"   {i}. Product ID: {match['product_id']} (Score: {match['score']:.1%})")
                        logger.info(f"      Text: {match['candidate_text'][:100]}...")
                    
                    logger.info(f"🎯 User Options:")
                    logger.info(f"   1. SAVE_NEW_PRODUCT - Save as completely new product")
                    logger.info(f"   2. EXISTING_PRODUCT - Add to existing product (auto-handles batch logic)")
                    
                    # Simulate user decision (in real implementation, this would be user input)
                    user_decision = simulate_user_decision(top_matches, validated_data)
                    
                    if user_decision["action"] == "SAVE_NEW_PRODUCT":
                        logger.info(f"🔄 USER DECISION: Save as new product")
                        try:
                            new_product_id = save_new_product_to_database(validated_data)
                            validated_data["saved_product_id"] = new_product_id
                            validated_data["user_intervention_resolved"] = "SAVE_NEW_PRODUCT"
                            logger.info(f"✅ User intervention resolved - Product saved with ID: {new_product_id}")
                            
                            # BM25 index is already updated by save_new_product_to_database function
                        except Exception as e:
                            logger.error(f"❌ Failed to save product after user intervention: {e}")
                            validated_data["user_intervention_error"] = str(e)
                    
                    elif user_decision["action"] == "EXISTING_PRODUCT":
                        logger.info(f"🔄 USER DECISION: Add to existing product - auto-handling batch logic")
                        target_product_id = user_decision["target_product_id"]
                        batch_number = validated_data.get(BATCH_NUMBER_FIELD, "")
                        quantity = validated_data.get(QUANTITY_FIELD, "")
                        
                        # Get batch number from existing product
                        existing_batch = _get_batch_number_from_database(target_product_id)
                        batch_match = batch_number and existing_batch and batch_number.lower() == existing_batch.lower()
                        
                        logger.info(f"🔍 AUTOMATED BATCH ANALYSIS:")
                        logger.info(f"   Current batch: '{batch_number}'")
                        logger.info(f"   Existing batch: '{existing_batch}'")
                        logger.info(f"   Batch match: {batch_match}")
                        
                        if batch_match:
                            # Same batch: Add quantity to existing
                            logger.info(f"✅ Same batch detected - adding quantity to existing product")
                            if not quantity or quantity.strip() == "":
                                quantity = "1"
                                logger.info(f"⚠️ No quantity specified, defaulting to 1")
                            
                            success = add_quantity_to_existing_product(target_product_id, quantity, batch_number)
                            if success:
                                validated_data["quantity_added"] = True
                                validated_data["target_product_id"] = target_product_id
                                validated_data["user_intervention_resolved"] = "EXISTING_PRODUCT_SAME_BATCH"
                                logger.info(f"✅ User intervention resolved - Added {quantity} to product {target_product_id}")
                            else:
                                logger.error(f"❌ Failed to add quantity to product {target_product_id}")
                                validated_data["user_intervention_error"] = "Failed to add quantity"
                        else:
                            # Different batch: Add new batch entry
                            logger.info(f"✅ Different batch detected - adding new batch entry")
                            success = add_new_batch_to_existing_product(target_product_id, validated_data)
                            if success:
                                validated_data["new_batch_added"] = True
                                validated_data["target_product_id"] = target_product_id
                                validated_data["user_intervention_resolved"] = "EXISTING_PRODUCT_NEW_BATCH"
                                logger.info(f"✅ User intervention resolved - Added new batch to product {target_product_id}")
                            else:
                                logger.error(f"❌ Failed to add new batch to product {target_product_id}")
                                validated_data["user_intervention_error"] = "Failed to add new batch"
                    
                    else:
                        logger.error(f"❌ Invalid user decision: {user_decision}")
                        validated_data["user_intervention_error"] = "Invalid user decision"
                
                # Handle automatic actions (high confidence scenarios)
                if duplicate_result["action_type"] in ["ADD_QUANTITY_EXISTING", "ADD_NEW_BATCH"]:
                    logger.info(f"🔄 AUTO-PROCESSING: {duplicate_result['action_reason']}")
                    
                    if duplicate_result["action_type"] == "ADD_QUANTITY_EXISTING":
                        # Add quantity to existing product
                        target_product_id = duplicate_result["product_id"]
                        batch_number = duplicate_result["batch_number"]
                        quantity = duplicate_result["quantity"]
                        
                        # Handle empty quantity
                        if not quantity or quantity.strip() == "":
                            quantity = "1"  # Default to 1 if no quantity specified
                            logger.info(f"⚠️ No quantity specified, defaulting to 1")
                        
                        success = add_quantity_to_existing_product(target_product_id, quantity, batch_number)
                        if success:
                            validated_data["quantity_added"] = True
                            validated_data["target_product_id"] = target_product_id
                            logger.info(f"✅ Successfully added {quantity} to product {target_product_id}")
                        else:
                            validated_data["quantity_add_error"] = "Failed to add quantity"
                            logger.error(f"❌ Failed to add quantity to product {target_product_id}")
                    
                    elif duplicate_result["action_type"] == "ADD_NEW_BATCH":
                        # Add new batch to existing product
                        target_product_id = duplicate_result["product_id"]
                        new_batch_data = {
                            "BatchNumber": duplicate_result["batch_number"],
                            "UnitSellingPrice": validated_data.get("UnitSellingPrice"),
                            "ManufacturingDate": validated_data.get("ManufacturingDate"),
                            "ExpiryDate": validated_data.get("ExpiryDate")
                        }
                        
                        success = add_new_batch_to_existing_product(target_product_id, new_batch_data)
                        if success:
                            validated_data["new_batch_added"] = True
                            validated_data["target_product_id"] = target_product_id
                            logger.info(f"✅ Successfully added new batch to product {target_product_id}")
                        else:
                            validated_data["new_batch_add_error"] = "Failed to add new batch"
                            logger.error(f"❌ Failed to add new batch to product {target_product_id}")
                elif duplicate_result["action_type"] == "SAVE_NEW_PRODUCT":
                    logger.info(f"💾 SAVING NEW: {duplicate_result['action_reason']}")
                    try:
                        # Save the new product to database
                        new_product_id = save_new_product_to_database(validated_data)
                        validated_data["saved_product_id"] = new_product_id
                        logger.info(f"✅ Product successfully saved with ID: {new_product_id}")
                        
                        # BM25 index is already updated by save_new_product_to_database function
                    except Exception as e:
                        logger.error(f"❌ Failed to save product to database: {e}")
                        validated_data["save_error"] = str(e)
                else:
                    logger.info(f"💾 SAVING NEW: {duplicate_result['action_reason']}")
                    try:
                        # Save the new product to database
                        new_product_id = save_new_product_to_database(validated_data)
                        validated_data["saved_product_id"] = new_product_id
                        logger.info(f"✅ Product successfully saved with ID: {new_product_id}")
                        
                        # BM25 index is already updated by save_new_product_to_database function
                    except Exception as e:
                        logger.error(f"❌ Failed to save product to database: {e}")
                        validated_data["save_error"] = str(e)
                
                # Also keep the old embedding for BriefDescription for backward compatibility
                brief_desc = validated_data.get("BriefDescription", "")
                if brief_desc:
                    validated_data["EmbeddedProductDescription"] = get_openai_embedding(brief_desc)
                else:
                    validated_data["EmbeddedProductDescription"] = []
            else:
                logger.info("⏭️ Embedding generation disabled or no internaldescription available")
                validated_data["InternalDescriptionVector"] = []
                validated_data["EmbeddedProductDescription"] = []
                validated_data["DuplicateDetection"] = {
                    "is_duplicate": False,
                    "is_new_product": True,
                    "top_matches": [],
                    "confidence": "unknown",
                    "reason": "Embedding generation disabled"
                }
            
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(validated_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Final structured invoice with validation and embedding written to: {final_path}")
        except Exception as e:
            logger.error(f"Error parsing or saving JSON with validation: {e}")
            # Fallback: save the raw output
            with open(final_path, "w", encoding="utf-8") as f:
                f.write(final_output)
        logger.info(f"Merged OCR text written to: {merged_path}")
        return final_output
    except openai.APIError as e:
        if "rate_limit" in str(e).lower() or "quota" in str(e).lower():
            logger.error(f"OpenAI API rate limit exceeded: {e}")
        else:
            logger.error(f"OpenAI API error: {e}")
        raise  # Let retry decorator handle this
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API rate limit exceeded: {e}")
        raise  # Let retry decorator handle this
    except Exception as e:
        logger.error(f"Unexpected error sending to OpenAI LLM: {e}")
        raise


def main():
    """
    Main entry point for the OCR and inventory extraction pipeline.
    Runs the full pipeline: OCR, LLM extraction, internaldescription creation, embedding, hybrid search, and output.
    Steps:
        1. Runs OCR and Vision processing concurrently for better performance.
        2. Uses intelligent source selection to choose best data source for each field.
        3. Sends merged OCR text and Vision data to the LLM for extraction.
        4. Creates internaldescription field by concatenating key product information.
        5. Generates embeddings for internaldescription (512 dimensions) if enabled.
        6. Performs hybrid BM25+cosine search for duplicate detection against existing products.
        7. Saves and prints the final structured output with embeddings and duplicate detection results.
    """
    start_time = time.time()
    logger.info("🚀 Starting OCR and inventory extraction pipeline...")
    
    # Step 1: Concurrent OCR and Vision Processing
    logger.info("🔄 Starting concurrent OCR and Vision processing...")
    
    # Prepare image files for Vision processing
    image_files = [
        os.path.join(FOLDER_PATH, f)
        for f in os.listdir(FOLDER_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    
    # Run OCR and Vision concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit OCR task
        ocr_future = executor.submit(process_folder, FOLDER_PATH)
        
        # Submit Vision task
        vision_future = executor.submit(aggregate_vision_fields_batch, image_files, [
            'Name', 'TradeName', 'Brand', 'MajorActiveIngredients', 'BarCodeNumber', 
            'BatchNumber', 'ManufacturingDate', 'ExpiryDate', 'UnitSellingPrice', 
            'AdministeredUOM', 'AdministeredTotalUnits'
        ], batch_size=3)
        
        # Wait for both to complete and measure timing
        ocr_start = time.time()
        merged_ocr_text = ocr_future.result()
        ocr_duration = time.time() - ocr_start
        
        vision_start = time.time()
        vision_fields = vision_future.result()
        vision_duration = time.time() - vision_start
        
        # Calculate concurrent processing time
        concurrent_duration = max(ocr_duration, vision_duration)
    
    logger.info(f"✅ Concurrent OCR processing completed in {ocr_duration:.2f}s")
    logger.info(f"✅ Concurrent Vision processing completed in {vision_duration:.2f}s")
    logger.info(f"✅ Total concurrent processing time: {concurrent_duration:.2f}s")
    
    # Step 2: LLM Extraction with Intelligent Source Selection
    llm_start = time.time()
    logger.info(f"🤖 Sending merged OCR text and Vision data to {INVENTORY_LLM_PROVIDER.upper()} LLM with intelligent source selection...")
    final_output = send_to_llm(merged_ocr_text, FOLDER_PATH, vision_fields)
    llm_duration = time.time() - llm_start
    logger.info(f"✅ LLM extraction completed in {llm_duration:.2f}s")
    
    # Final timing summary
    total_duration = time.time() - start_time
    logger.info("🎉 Pipeline completed successfully!")
    logger.info(f"📊 Performance Summary:")
    logger.info(f"   • OCR Processing: {ocr_duration:.2f}s")
    logger.info(f"   • Vision Processing: {vision_duration:.2f}s")
    logger.info(f"   • Concurrent Processing Time: {concurrent_duration:.2f}s")
    logger.info(f"   • LLM Extraction: {llm_duration:.2f}s")
    logger.info(f"   • Total Pipeline: {total_duration:.2f}s")
    
    print("\n----- FINAL STRUCTURED OUTPUT WITH INTELLIGENT SOURCE SELECTION -----\n")
    print(final_output)

if __name__ == "__main__":
    main()
