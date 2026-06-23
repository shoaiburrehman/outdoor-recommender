import optuna
import numpy as np
import pandas as pd
import wandb
from src.recommender import HybridRecommender

class Evaluator:
    def __init__(self):
        self.recommender = HybridRecommender()
        self.df = self.recommender.products_df

    def evaluate_precision_at_k(self, k=5, w_text=0.33, w_image=0.33, w_meta=0.33):
        precision_scores = []
        sample_size = min(50, len(self.df))
        test_indices = np.random.choice(len(self.df), size=sample_size, replace=False)

        for idx in test_indices:
            target_cat = self.df.iloc[idx].get('category', 'unknown')
            target_brand = self.df.iloc[idx]['brand']
            
            recs, _ = self.recommender.get_recommendations(
                target_idx=idx, top_k=k, w_text=w_text, w_image=w_image, w_meta=w_meta
            )
            
            hits = recs.apply(lambda r: r.get('category') == target_cat or r['brand'] == target_brand, axis=1).sum()
            precision_scores.append(hits / k)

        return np.mean(precision_scores)

def optimize_hyperparameters():
    # Initialize a clean Wandb run to visually track optimization metrics for your prof
    wandb.init(project="outdoor-recommender-tuning", entity=None, job_type="hyperparameter-tuning")
    
    evaluator = Evaluator()

    def objective(trial):
        w_text = trial.suggest_float('w_text', 0.0, 1.0)
        w_image = trial.suggest_float('w_image', 0.0, 1.0)
        w_meta = trial.suggest_float('w_meta', 0.0, 1.0)
        
        score = evaluator.evaluate_precision_at_k(k=5, w_text=w_text, w_image=w_image, w_meta=w_meta)
        
        # Log metrics onto active dashboard
        wandb.log({
            "trial_number": trial.number,
            "weight_text": w_text,
            "weight_image": w_image,
            "weight_metadata": w_meta,
            "precision_at_5": score
        })
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    # Log the overall absolute best configurations
    wandb.config.update(study.best_params)
    wandb.log({"best_precision_at_5": study.best_value})
    wandb.finish()
    
    print(f"\nOptimization Finished! Best Score: {study.best_value:.4f}")
    return study.best_params

if __name__ == "__main__":
    optimize_hyperparameters()