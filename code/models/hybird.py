import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader

class HybridRecommender:
    def __init__(self, cb_model, svd_model, ratings_df, alpha=0.5):
        self.cbf = cb_model
        self.svd = svd_model
        self.ratings_df = ratings_df
        self.alpha = alpha

    def recommend(self, user_id, seen_places, top_n=10):
        cbf_recs = self.cbf.recommend(user_id, seen_places=set(seen_places), top_n=100)
        cbf_scores = {place_id: score for place_id, score in cbf_recs}

        all_place_ids = self.cbf.places_df['Place_Id'].tolist()
        unseen_place_ids = [pid for pid in all_place_ids if pid not in seen_places]

        hybrid_scores = []
        for pid in unseen_place_ids:
            try:
                svd_score = self.svd.predict(str(user_id), str(pid)).est
                cbf_score = cbf_scores.get(pid, 0.0)
                final_score = self.alpha * svd_score + (1 - self.alpha) * cbf_score
                hybrid_scores.append((pid, final_score))
            except:
                continue

        print(f"[DEBUG] SVD Score: {svd_score:.2f}, CBF Score: {cbf_score:.2f}, Final: {final_score:.2f}")

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:top_n]
