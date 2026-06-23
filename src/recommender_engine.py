import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ENG_DATA_FILE = "data/products_engineered.csv"
SIMILARITY_FILE = "data/similarity_matrix.npy"
EMBEDDING_FILE = "data/embeddings.npy"

class OutdoorRecommender:
    def __init__(self):
        print("⏳ Initializing Hybrid Recommender Engine Artifacts...")
        self.df = pd.read_csv(ENG_DATA_FILE)
        self.similarity_matrix = np.load(SIMILARITY_FILE)
        self.embeddings = np.load(EMBEDDING_FILE)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("🎉 Hybrid Engine active! Balanced sorting logic online.\n")

    def get_hybrid_recommendations(self, product_name, top_n=5, w_sim=0.70, w_rating=0.20, w_discount=0.10):
        """
        Finds similar items by balancing text similarity coordinates with structural metrics 
        like rating scores and discounts (Hybrid Recommendation).
        """
        matches = self.df[self.df['name'].str.lower() == product_name.lower()]
        if matches.empty:
            print(f"❌ Product '{product_name}' not found in the catalog matrix.")
            return pd.DataFrame()
            
        product_idx = matches.index[0]
        
        # Pull the base text similarities for this specific item
        base_similarities = self.similarity_matrix[product_idx]
        
        # Normalize structural signals between 0.0 and 1.0 for equitable math balancing
        max_rating = 5.0
        max_discount = self.df['discount_percentage'].max() if self.df['discount_percentage'].max() > 0 else 1.0
        
        normalized_ratings = self.df['rating_score'] / max_rating
        normalized_discounts = self.df['discount_percentage'] / max_discount
        
        # Compute the integrated hybrid matrix equation row-by-row
        hybrid_scores = (w_sim * base_similarities) + (w_rating * normalized_ratings) + (w_discount * normalized_discounts)
        
        # Package scores into list entries to sort them cleanly
        scored_items = list(enumerate(hybrid_scores))
        sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        
        # Filter out the queried item itself from taking top slot
        top_indices = [item[0] for item in sorted_items if item[0] != product_idx][:top_n]
        
        return self.df.iloc[top_indices][['name', 'brand', 'price_clean', 'category', 'rating_score', 'discount_percentage']]

    def hybrid_semantic_search(self, query, top_n=5, w_sim=0.70, w_rating=0.20, w_discount=0.10):
        """Converts raw user queries into semantic coordinates and applies the scoring layers."""
        print(f"🔍 Executing hybrid vector scan for: '{query}'")
        query_vector = self.model.encode([query])
        text_similarities = cosine_similarity(query_vector, self.embeddings).flatten()
        
        max_rating = 5.0
        max_discount = self.df['discount_percentage'].max() if self.df['discount_percentage'].max() > 0 else 1.0
        
        normalized_ratings = self.df['rating_score'] / max_rating
        normalized_discounts = self.df['discount_percentage'] / max_discount
        
        # Compute multi-attribute score integration
        hybrid_scores = (w_sim * text_similarities) + (w_rating * normalized_ratings) + (w_discount * normalized_discounts)
        
        top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
        return self.df.iloc[top_indices][['name', 'brand', 'price_clean', 'category', 'rating_score', 'discount_percentage']]

if __name__ == "__main__":
    engine = OutdoorRecommender()
    test_item = "KalmarSt. 3L Rain Jacket II"
    
    print(f"--- 🏷️ HYBRID BALANCED ITEMS SIMILAR TO: '{test_item}' ---")
    recs = engine.get_hybrid_recommendations(test_item, top_n=5)
    print(recs.to_string(), "\n")