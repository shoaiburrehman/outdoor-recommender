import os
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_FILE = "data/embeddings.npy"
SIMILARITY_FILE = "data/similarity_matrix.npy"

def calculate_similarity_matrix():
    if not os.path.exists(EMBEDDING_FILE):
        print(f"❌ Error: {EMBEDDING_FILE} not found. Please run Step 2 first.")
        return

    print("⏳ Loading high-density vector embeddings...")
    embeddings = np.load(EMBEDDING_FILE)
    print(f"📊 Loaded vector matrix of shape: {embeddings.shape}")

    print("🧮 Calculating Cosine Similarity Matrix cross-products...")
    # This computes a 2222 x 2222 matrix of similarity scores between 0.0 and 1.0
    similarity_matrix = cosine_similarity(embeddings)
    print(f"📐 Similarity matrix built successfully with shape: {similarity_matrix.shape}")

    # Persist the calculated matrix to disk
    os.makedirs("data", exist_ok=True)
    np.save(SIMILARITY_FILE, similarity_matrix)
    print(f"🎉 Success! Similarity matrix binary saved directly to {SIMILARITY_FILE}")

if __name__ == "__main__":
    calculate_similarity_matrix()