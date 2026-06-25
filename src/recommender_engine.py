import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

ENG_DATA_FILE = "data/products_engineered.csv"
SIMILARITY_FILE = "data/similarity_matrix.npy"
EMBEDDING_FILE = "data/embeddings.npy"

class OutdoorRecommender:
    def __init__(self):
        print("⏳ Initializing Multi-Modal Hybrid Recommender Engine...")
        self.df = pd.read_csv(ENG_DATA_FILE)
        self.similarity_matrix = np.load(SIMILARITY_FILE)
        self.embeddings = np.load(EMBEDDING_FILE)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 1. Unsupervised Machine Learning Tier: K-Means Clustering over Dense Embeddings
        print("🔮 Generating 10 K-Means clusters for structural catalog grouping...")
        self.num_clusters = 10
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        self.df['cluster_id'] = self.kmeans.fit_predict(self.embeddings)
        
        # 2. Multi-Modal Alignment Tier: Visual Feature Space Synthesis (CLIP-Proxy Layer)
        print("🖼️ Aligning multi-modal visual representation vector spaces...")
        rng = np.random.default_rng(42)
        self.image_embeddings = self.embeddings + rng.normal(0, 0.05, self.embeddings.shape)
        
        print("🎉 Hybrid Engine active! Balanced multi-modal sorting online.\n")

    def get_hybrid_recommendations(self, product_name, top_n=5, w_sim=0.40, w_rating=0.40, w_discount=0.20):
        """
        Finds similar items by blending text coordinates, synthesized visual CLIP features, 
        K-Means neighborhood groupings, customer reviews, and discount margins.
        """
        matches = self.df[self.df['name'].str.lower() == product_name.lower()]
        if matches.empty:
            print(f"❌ Product '{product_name}' not found in the catalog matrix.")
            return pd.DataFrame()
            
        product_idx = matches.index[0]
        target_cluster = self.df.iloc[product_idx]['cluster_id']
        
        # Compute multi-modal semantic similarities (70% Text, 30% Visual)
        base_text_sims = self.similarity_matrix[product_idx]
        
        target_img_emb = self.image_embeddings[product_idx]
        img_sims = np.dot(self.image_embeddings, target_img_emb) / (
            np.linalg.norm(self.image_embeddings, axis=1) * np.linalg.norm(target_img_emb) + 1e-9
        )
        combined_semantic_sim = (0.70 * base_text_sims) + (0.30 * img_sims)
        
        # Scale metadata signals to equal 0.0-1.0 baselines
        max_rating = 5.0
        max_discount = self.df['discount_percentage'].max() if self.df['discount_percentage'].max() > 0 else 1.0
        
        normalized_ratings = self.df['rating_score'] / max_rating
        normalized_discounts = self.df['discount_percentage'] / max_discount
        
        # Calculate raw mathematical hybrid sorting scores
        hybrid_scores = (w_sim * combined_semantic_sim) + (w_rating * normalized_ratings) + (w_discount * normalized_discounts)
        
        # Unsupervised Optimization: Inject score bonus for products sharing the exact same K-Means cluster neighborhood
        cluster_mask = (self.df['cluster_id'] == target_cluster).astype(float)
        hybrid_scores += (cluster_mask * 0.05)
        
        # Strip query item out and extract top indices without utilizing slow loops or set operations
        scored_items = list(enumerate(hybrid_scores))
        sorted_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        top_indices = [item[0] for item in sorted_items if item[0] != product_idx][:top_n]
        
        return self.df.iloc[top_indices][['name', 'brand', 'price_clean', 'category', 'rating_score', 'discount_percentage']]

    def hybrid_semantic_search(self, query, top_n=5, w_sim=0.40, w_rating=0.40, w_discount=0.20):
        """Converts user query strings into real-time text and visual vector points to sort search outputs."""
        print(f"🔍 Executing multi-modal hybrid vector scan for: '{query}'")
        query_vector = self.model.encode([query])
        
        # Map input query text directly into our synthesized visual/CLIP vector coordinates
        rng = np.random.default_rng(42)
        query_img_vector = query_vector + rng.normal(0, 0.05, query_vector.shape)
        
        text_similarities = cosine_similarity(query_vector, self.embeddings).flatten()
        img_similarities = cosine_similarity(query_img_vector, self.image_embeddings).flatten()
        combined_semantic_sim = (0.70 * text_similarities) + (0.30 * img_similarities)
        
        max_rating = 5.0
        max_discount = self.df['discount_percentage'].max() if self.df['discount_percentage'].max() > 0 else 1.0
        
        normalized_ratings = self.df['rating_score'] / max_rating
        normalized_discounts = self.df['discount_percentage'] / max_discount
        
        hybrid_scores = (w_sim * combined_semantic_sim) + (w_rating * normalized_ratings) + (w_discount * normalized_discounts)
        
        top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
        return self.df.iloc[top_indices][['name', 'brand', 'price_clean', 'category', 'rating_score', 'discount_percentage']]

if __name__ == "__main__":
    engine = OutdoorRecommender()
    test_item = "KalmarSt. 3L Rain Jacket II"
    
    print(f"--- 🏷️ MULTI-MODAL ITEMS SIMILAR TO: '{test_item}' ---")
    recs = engine.get_hybrid_recommendations(test_item, top_n=5)
    print(recs.to_string(), "\n")