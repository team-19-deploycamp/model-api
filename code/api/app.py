from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import random
import pickle
import os
import sys  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.getData import load_ratings, load_places

# Load model
with open("../models/cbf_v2.pkl", "rb") as f:
    model = pickle.load(f)

# Pastikan model memiliki ratings_df & places_df
ratings_df = model.ratings_df
places_df = model.places_df
users_df = model.users_df

app = FastAPI()

# ======== Request/Response Schema ============
class ColdStartRequest(BaseModel):
    selected_place_ids: list[int]
    top_n: int = 10

class Preference(BaseModel):
    Place_Id: int
    Rating: float

class PreferenceSubmission(BaseModel):
    User_Id: int
    Preferences: List[Preference]

# ========== GET 10 tempat acak ============
@app.get("/cold_start")
def get_cold_start_places():
    categories = places_df['Category'].dropna().unique()
    selected_places = pd.DataFrame()

    for category in categories:
        subset = places_df[places_df['Category'] == category]
        if not subset.empty:
            sampled = subset.sample(n=1, random_state=random.randint(1, 999))
            selected_places = pd.concat([selected_places, sampled])

    remaining = 10 - len(selected_places)
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
def submit_user_preferences(data: PreferenceSubmission):
    user_id = data.User_Id
    prefs = data.Preferences

    if not prefs:
        raise HTTPException(status_code=400, detail="Preferensi kosong.")

    seen_places = set([p.Place_Id for p in prefs])
    ratings_data = {
        "User_Id": [user_id] * len(prefs),
        "Place_Id": [p.Place_Id for p in prefs],
        "Place_Ratings": [p.Rating for p in prefs]
    }

    import pandas as pd
    temp_df = pd.DataFrame(ratings_data)

    # Buat model khusus untuk user baru
    from models.content_based import ContentBasedRecommender
    cbf = ContentBasedRecommender(places_df, temp_df, users_df)
    cbf.fit(temp_df)

    recommendations = cbf.recommend(user_id, seen_places=seen_places, top_n=10)

    result = []
    for place_id, score in recommendations:
        place = places_df[places_df['Place_Id'] == place_id].iloc[0]
        result.append({
            "Place_Id": int(place_id),
            "Place_Name": place['Place_Name'],
            "Category": place['Category'],
            "City": place['City'],
            "Score": round(score, 4)
        })

    return {
        "User_Id": user_id,
        "Recommendations": result
    }
