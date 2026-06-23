import sys
import os

# Ensure local source directory parameters are append-bound cleanly to active runtime context
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.scraper import run_scraper
from src.embeddings import EmbeddingPipeline
from src.evaluate import optimize_hyperparameters

def main():
    print("=====================================================================")
    print("      LAUNCHING END-TO-END OUTDOOR RECOMMENDER PIPELINE APP          ")
    print("=====================================================================\n")

    # Step 1: Broad Scale Multi-Category Web Ingestion Execution
    print("[STAGE 1/3] Launching Intelligent Broad Target Crawler Engine...")
    # Using 3 pages per category for a rapid initial test drive of the asset pipeline
    run_scraper(max_pages_per_category=3)

    # Step 2: Computer Vision & Natural Language Context Embedding Production
    print("\n[STAGE 2/3] Extracting Multimodal Vectors via CLIP & Scaling Structural Metadata...")
    embedder = EmbeddingPipeline()
    embedder.generate_embeddings()

    # Step 3: Algorithmic Calibration via Metric Evaluation Loops
    print("\n[STAGE 3/3] Commencing Multi-Trial Optimization Sweep via Optuna Algorithms...")
    best_weights = optimize_hyperparameters()

    print("\n=====================================================================")
    print("  PIPELINE COMPLETE: Your dataset is scaled, vector fields are written, ")
    print("  and your model's hyperparameters are optimized. Ready for Streamlit UI! ")
    print("=====================================================================")

if __name__ == "__main__":
    main()