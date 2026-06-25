import unittest
import os
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from src.recommender_engine import OutdoorRecommender

class TestHybridOutdoorRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initializes the backend recommender artifacts once."""
        cls.recommender = OutdoorRecommender()

    def test_matrix_and_embedding_dimensions(self):
        """Ensure all text and visual embedding spaces perfectly align with the catalog shape."""
        row_count = len(self.recommender.df)
        
        # Validate base similarity matrices and text coordinates
        self.assertEqual(self.recommender.similarity_matrix.shape, (row_count, row_count))
        self.assertEqual(self.recommender.embeddings.shape, (row_count, 384))
        
        # UPDATED: Assert structural integrity of the multi-modal image embedding space
        self.assertEqual(self.recommender.image_embeddings.shape, (row_count, 384))

    def test_kmeans_clustering_integrity(self):
        """Verify that the unsupervised K-Means layer successfully partitioned the product data."""
        df = self.recommender.df
        
        # Assert cluster column exists in the DataFrame
        self.assertIn('cluster_id', df.columns)
        
        # Assert that cluster assignments fall perfectly within the configured K=10 boundaries
        unique_clusters = df['cluster_id'].unique()
        self.assertTrue(all(0 <= cid < 10 for cid in unique_clusters))
        
        # Check that clustering isn't empty or corrupted
        self.assertGreater(len(unique_clusters), 1)

    def test_hybrid_recommendations_structure(self):
        """Verify that the hybrid recommender factors in all tracking metrics and schema correctly."""
        test_item = "KalmarSt. 3L Rain Jacket II"
        recs = self.recommender.get_hybrid_recommendations(test_item, top_n=5)
        
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 5)
        self.assertIn('rating_score', recs.columns)
        self.assertIn('discount_percentage', recs.columns)
        self.assertIn('category', recs.columns)

    def test_hybrid_semantic_search_execution(self):
        """Validate that raw text queries correctly map to the dual text/visual vector layers."""
        query = "waterproof hiking boots for winter"
        search_results = self.recommender.hybrid_semantic_search(query, top_n=3)
        
        self.assertIsInstance(search_results, pd.DataFrame)
        self.assertEqual(len(search_results), 3)
        self.assertIn('name', search_results.columns)
        self.assertIn('brand', search_results.columns)

    def test_hybrid_scoring_influence(self):
        """Verify that high structural metadata coefficients alter the baseline vector sorting order."""
        test_item = "KalmarSt. 3L Rain Jacket II"
        
        # Scenario A: Rely strictly on semantic features
        pure_semantic_recs = self.recommender.get_hybrid_recommendations(
            test_item, top_n=5, w_sim=1.0, w_rating=0.0, w_discount=0.0
        )
        
        # Scenario B: High emphasis on business drivers (ratings and active store discounts)
        hybrid_recs = self.recommender.get_hybrid_recommendations(
            test_item, top_n=5, w_sim=0.4, w_rating=0.4, w_discount=0.2
        )
        
        # Validation Pass Criteria: Product names list must change due to hybrid ranking re-sorting
        self.assertNotEqual(pure_semantic_recs['name'].tolist(), hybrid_recs['name'].tolist())

if __name__ == "__main__":
    unittest.main()