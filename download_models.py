import os
import sys

print("Initializing SentenceTransformer downloader...")
try:
    from sentence_transformers import SentenceTransformer
    model_name = "microsoft/harrier-oss-v1-0.6b"
    print(f"Downloading weights for model '{model_name}'...")
    model = SentenceTransformer(model_name)
    print("Download complete. Model weights cached successfully in Hugging Face directory.")
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
