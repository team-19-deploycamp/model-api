from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import random
import pickle
import os
import sys  
import math
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.getData import load_ratings, load_places, load_users

# Load models
with open("../models/SVD.pkl", "rb") as f:   
    svd_model = pickle.load(f)

with open("../models/cbf_v2.pkl", "rb") as f:   
    cbf_model = pickle.load(f)

# Load XGBoost meta model
meta_model = xgb.Booster()
meta_model.load_model("../models/hybrid_meta_xgb.json")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Hybrid model class
class HybridRecommenderWithXGB:
    def __init__(self, svd_model, cbf_model, meta_model):
        self.svd = svd_model
        self.cbf = cbf_model
        self.meta_model = meta_model

    def predict(self, user_id, place_id):
        est_svd = self.svd.predict(user_id, place_id).est
        est_cbf = self.cbf.predict_single(user_id, place_id)
        dtest = xgb.DMatrix([[est_svd, est_cbf]])
        pred = self.meta_model.predict(dtest)[0]
        return pred

    def recommend(self, user_id, seen_places, top_n=20):
        if user_id not in self.cbf.user_profiles:
            # User baru fallback ke CBF saja
            return self.cbf.recommend(user_id, seen_places, top_n=top_n)

        scores = []
        for place_id in self.cbf.places_df['Place_Id']:
            if place_id not in seen_places:
                try:
                    score = self.predict(user_id, place_id)
                except Exception:
                    score = 3.0  # fallback skor
                scores.append((place_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

# Load data
data, ratings_df = load_ratings()
_, places_df = load_places()
_, users_df = load_users()

# Fit CBF user profiles (penting supaya rekomendasi bisa jalan)
cbf_model.fit(ratings_df)

# Instantiate hybrid model
hybrid_model = HybridRecommenderWithXGB(svd_model, cbf_model, meta_model)

app = FastAPI()

# ======== Request/Response Schema ============
class ColdStartRequest(BaseModel):
    selected_place_ids: list[int]
    top_n: int = 20

class Preference(BaseModel):
    Place_Id: int
    Rating: float

class PreferenceSubmission(BaseModel):
    User_Id: int
    Latitude: float
    Longitude: float
    Preferences: List[Preference]

# ========== GET 20 tempat acak ============
@app.get("/cold_start")
def get_cold_start_places():
    categories = places_df['Category'].dropna().unique()
    selected_places = pd.DataFrame()

    for category in categories:
        subset = places_df[places_df['Category'] == category]
        if not subset.empty:
            sampled = subset.sample(n=1, random_state=random.randint(1, 999))
            selected_places = pd.concat([selected_places, sampled])

    remaining = 20 - len(selected_places)
    if remaining > 0:   
        remaining_pool = places_df[~places_df['Place_Id'].isin(selected_places['Place_Id'])]
        extra = remaining_pool.sample(n=remaining, random_state=random.randint(1, 999))
        selected_places = pd.concat([selected_places, extra])

    selected_places = selected_places.sample(frac=1).reset_index(drop=True)

    result = []
    for _, row in selected_places.iterrows():
        result.append({
            "Place_Id": int(row['Place_Id']),
            "Place_Name": row['Place_Name'],
            "Category": row['Category'],
            "City": row['City'],
            "Rating": None  # frontend mengisi rating user
        })

    if not result:
        raise HTTPException(status_code=404, detail="Tidak ada tempat ditemukan.")

    return result

# ========== POST rekomendasi dari input user baru ============
@app.post("/submit_preference")
def submit_user_preferences(data: PreferenceSubmission, max_distance_km: float = 10):
    user_id = data.User_Id
    user_lat = data.Latitude
    user_lon = data.Longitude
    prefs = data.Preferences

    if not prefs:
        raise HTTPException(status_code=400, detail="Preferensi kosong.")

    seen_places = set([p.Place_Id for p in prefs])
    ratings_data = {
        "User_Id": [user_id] * len(prefs),
        "Place_Id": [p.Place_Id for p in prefs],
        "Place_Ratings": [p.Rating for p in prefs]
    }

    temp_df = pd.DataFrame(ratings_data)

    # Update CBF dengan data preferensi user baru
    cbf_new = type(cbf_model)(places_df, temp_df, users_df)
    cbf_new.fit(temp_df)

    # Update hybrid model dengan CBF baru untuk user baru
    hybrid_for_new_user = HybridRecommenderWithXGB(svd_model, cbf_new, meta_model)

    recommendations = hybrid_for_new_user.recommend(user_id, seen_places=seen_places, top_n=20)

    result = []
    for place_id, score in recommendations:
        place = places_df[places_df['Place_Id'] == place_id].iloc[0]
        distance = haversine(user_lat, user_lon, place['Lat'], place['Long'])
        
        result.append({
            "Place_Id": int(place_id),
            "Place_Name": place['Place_Name'],
            "Category": place['Category'],
            "City": place['City'],
            "Distance_km": round(distance, 2),
            "Score": float(round(score, 4))
        })

    # ✅ Rerank: sort first by distance, then by score
    result.sort(key=lambda x: (x["Distance_km"], -x["Score"]))

    # Keep top 20 after reranking
    result = result[:20]

    return {
        "User_Id": user_id,
        "Recommendations": result
    }
