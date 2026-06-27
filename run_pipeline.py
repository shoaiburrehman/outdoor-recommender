import sys
import os

# Ensure local source directory parameters are append-bound cleanly to active runtime context
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import every file in the logical sequence
from src.scraper import run_scraper
from src.feature_engineering import generate_product_profiles
from src.embedding_generator import EmbeddingPipeline
from src.similarity_calculator import calculate_similarity_matrix  # ⚡ Added
from src.evaluate import optimize_hyperparameters

def main():
    print("=====================================================================")
    print("      LAUNCHING END-TO-END OUTDOOR RECOMMENDER PIPELINE APP          ")
    print("=====================================================================\n")

    # STAGE 1: Data Gathering
    print("[STAGE 1/5] Ingesting Web Data via Intelligent Crawler Engine...")
    run_scraper(max_pages_per_category=6)

    # STAGE 2: Data Processing & Feature Engineering
    print("\n[STAGE 2/5] Cleaning Text Elements & Structuring Metadata...")
    generate_product_profiles() 

    # STAGE 3: Dense Vector Space Production
    print("\n[STAGE 3/5] Computing High-Density Transformers Embeddings...")
    embedder = EmbeddingPipeline()
    embedder.generate_embeddings()

    # STAGE 4: Mathematical Comparison Mapping ⚡ THE CRITICAL SYNC LINK
    print("\n[STAGE 4/5] Building Unified Global Cosine Similarity Matrix...")
    calculate_similarity_matrix() 

    # STAGE 5: Hyperparameter Optimization Sweep
    print("\n[STAGE 5/5] Launching Optuna Parameter Loop & W&B Tracking...")
    best_weights = optimize_hyperparameters()

    print("\n=====================================================================")
    print("  PIPELINE COMPLETE: Your dataset is scaled, vector fields are written, ")
    print("  and your model's hyperparameters are optimized. Ready for Streamlit UI! ")
    print("=====================================================================")

if __name__ == "__main__":
    main()