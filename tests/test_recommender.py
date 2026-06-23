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

    def test_matrix_dimensions(self):
        """Ensure the dataset shape perfectly lines up with the vector dimensions."""
        row_count = len(self.recommender.df)
        self.assertEqual(self.recommender.similarity_matrix.shape, (row_count, row_count))
        self.assertEqual(self.recommender.embeddings.shape, (row_count, 384))

    def test_hybrid_recommendations_structure(self):
        """Verify that the hybrid recommender factors in all new columns correctly."""
        test_item = "KalmarSt. 3L Rain Jacket II"
        recs = self.recommender.get_hybrid_recommendations(test_item, top_n=5)
        
        self.assertIsInstance(recs, pd.DataFrame)
        self.assertEqual(len(recs), 5)
        # Check that your ranking metrics are contained in the output matrix
        self.assertIn('rating_score', recs.columns)
        self.assertIn('discount_percentage', recs.columns)

    def test_hybrid_scoring_influence(self):
        """Verify that high ratings/discounts alter the standard text sorting index."""
        test_item = "KalmarSt. 3L Rain Jacket II"
        
        # Scenario A: Rely strictly on text similarity (w_sim=1.0)
        pure_text_recs = self.recommender.get_hybrid_recommendations(test_item, top_n=5, w_sim=1.0, w_rating=0.0, w_discount=0.0)
        
        # Scenario B: High emphasis on customer reviews and discounts
        hybrid_recs = self.recommender.get_hybrid_recommendations(test_item, top_n=5, w_sim=0.4, w_rating=0.4, w_discount=0.2)
        
        # Pass Criteria: The top-recommended product list indices must change 
        # because ratings/discounts re-ordered the raw semantic similarities.
        self.assertNotEqual(pure_text_recs['name'].tolist(), hybrid_recs['name'].tolist())

if __name__ == "__main__":
    unittest.main()