# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import optuna
import wandb
import random
from src.recommender_engine import OutdoorRecommender
import json
import os

class Evaluator:
    def __init__(self):
        self.recommender = OutdoorRecommender()
        self.df = self.recommender.df

    def evaluate_metrics_at_k(self, k=5, w_sim=0.40, w_rating=0.40, w_discount=0.20):
        precision_scores = []
        recall_scores = []
        
        sample_size = min(50, len(self.df))
        np.random.seed(42)
        test_indices = np.random.choice(len(self.df), size=sample_size, replace=False)

        for idx in test_indices:
            target_cat = self.df.iloc[idx].get('category', 'unknown')
            target_brand = self.df.iloc[idx]['brand']
            target_name = self.df.iloc[idx]['name']
            
            total_relevant_in_db = len(self.df[(self.df['category'] == target_cat) & (self.df['brand'] == target_brand)])
            
            try:
                recs = self.recommender.get_hybrid_recommendations(target_name, top_n=k, w_sim=w_sim, w_rating=w_rating, w_discount=w_discount)
                if not recs.empty:
                    hits = recs.apply(lambda r: (r.get('category') == target_cat) & (r['brand'] == target_brand), axis=1).sum()
                    precision_scores.append(hits / k)
                    if total_relevant_in_db > 0:
                        recall_scores.append(hits / total_relevant_in_db)
                    else:
                        recall_scores.append(0.0)
            except Exception:
                pass

        return np.mean(precision_scores) if precision_scores else 0.0, np.mean(recall_scores) if recall_scores else 0.0

    def run_perturbation_analysis(self, best_params, k=5):
        print("\n⚡ Initializing Perturbation Analysis (Data Character Corruption Test)...")
        corrupted_precision_scores = []
        sample_size = min(30, len(self.df))
        test_indices = np.random.choice(len(self.df), size=sample_size, replace=False)

        for idx in test_indices:
            target_cat = self.df.iloc[idx].get('category', 'unknown')
            target_brand = self.df.iloc[idx]['brand']
            original_name = self.df.iloc[idx]['name']
            
            name_chars = list(original_name)
            for _ in range(max(1, len(name_chars) // 6)): 
                rand_pos = random.randint(0, len(name_chars) - 1)
                name_chars[rand_pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
            perturbed_name = "".join(name_chars)
            
            try:
                recs = self.recommender.hybrid_semantic_search(perturbed_name, top_n=k, **best_params)
                if not recs.empty:
                    hits = recs.apply(lambda r: (r.get('category') == target_cat) & (r['brand'] == target_brand), axis=1).sum()
                    corrupted_precision_scores.append(hits / k)
            except Exception:
                pass
                
        baseline_p, _ = self.evaluate_metrics_at_k(k=k, **best_params)
        perturbed_p = np.mean(corrupted_precision_scores) if corrupted_precision_scores else 0.0
        drop = baseline_p - perturbed_p
        
        print(f" -> Baseline Precision:  {baseline_p:.4f}")
        print(f" -> Perturbed Precision: {perturbed_p:.4f}")
        print(f" -> Robustness Margin Drop: -{drop:.4f}")
        return drop

def optimize_hyperparameters():
    wandb.init(project="outdoor-recommender-tuning", job_type="hyperparameter-tuning")
    evaluator = Evaluator()
    best_run_tracker = {"precision": 0.0, "recall": 0.0}

    def objective(trial):
        w_sim = trial.suggest_float('w_sim', 0.0, 1.0)
        w_rating = trial.suggest_float('w_rating', 0.0, 1.0)
        w_discount = trial.suggest_float('w_discount', 0.0, 1.0)
        
        prec, recall = evaluator.evaluate_metrics_at_k(k=5, w_sim=w_sim, w_rating=w_rating, w_discount=w_discount)
        
        if prec > best_run_tracker["precision"]:
            best_run_tracker["precision"] = prec
            best_run_tracker["recall"] = recall
            
        wandb.log({
            "trial_number": trial.number,
            "weight_sim": w_sim,
            "weight_rating": w_rating,
            "weight_discount": w_discount,
            "precision_at_5": prec,
            "recall_at_5": recall
        })
        return prec

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    wandb.config.update(study.best_params)
    wandb.log({"best_precision_at_5": study.best_value, "corresponding_recall_at_5": best_run_tracker["recall"]})
    wandb.finish()
    
    print("\n" + "="*50)
    print("🚀 HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best Engine Weights: {study.best_params}")
    print(f"Validated Precision@5:  {study.best_value:.4f}")
    print(f"Corresponding Recall@5: {best_run_tracker['recall']:.4f}")
    print("="*50)
    
# 💾 EXPLICIT ROOT FIX: Persist parameters locally for Streamlit to read!
    os.makedirs("data", exist_ok=True)
    with open("data/best_weights.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("🎉 Success! Optimal tuning parameters written directly to data/best_weights.json")

    evaluator.run_perturbation_analysis(study.best_params, k=5)
    return study.best_params

if __name__ == "__main__":
    optimize_hyperparameters()