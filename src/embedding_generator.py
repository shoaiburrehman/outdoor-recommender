import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/products_engineered.csv"
EMBEDDING_FILE = "data/embeddings.npy"

# Wrap it in the class structure that run_pipeline.py imports! ⚡
class EmbeddingPipeline:
    def __init__(self):
        pass

    def generate_embeddings(self):
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Error: {INPUT_FILE} not found. Please run your feature engineering script first.")
            return

        print("⏳ Loading engineered metadata profiles...")
        df = pd.read_csv(INPUT_FILE)
        
        # Isolate our engineered text profiles into a list
        text_profiles = df['metadata_profile'].astype(str).tolist()
        print(f"📋 Found {len(text_profiles)} text targets ready for mathematical encoding.")

        print("🤖 Initializing SentenceTransformer Model ('all-MiniLM-L6-v2')...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        print("⚡ Generating dense vector embeddings (this might take a minute on CPU)...")
        embeddings = model.encode(text_profiles, show_progress_bar=True, batch_size=32)
        
        # Convert vectors into a standard NumPy array matrix
        embeddings_matrix = np.array(embeddings)
        print(f"📊 Vector Matrix built successfully with shape: {embeddings_matrix.shape}")

        # Persist the calculated vector space matrix to disk
        os.makedirs("data", exist_ok=True)
        np.save(EMBEDDING_FILE, embeddings_matrix)
        print(f"🎉 Success! High-density binary vectors saved directly to {EMBEDDING_FILE}")

if __name__ == "__main__":
    pipeline = EmbeddingPipeline()
    pipeline.generate_embeddings()