from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from supabase import create_client, Client
from postgrest.exceptions import APIError
import pytz
import pandas as pd
import random
import pickle
import re
import os
import sys  
import math
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.getData import load_ratings, load_places, load_users

# Load .env dari folder project
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

@asynccontextmanager
async def lifespan(app: FastAPI):
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase credentials not found")

    app.state.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[DEBUG] Supabase client initialized")
    yield
    print("[DEBUG] Lifespan ended")

app = FastAPI(lifespan=lifespan)
# Load models
with open("../models/SVD.pkl", "rb") as f:   
    svd_model = pickle.load(f)

with open("../models/cbf_v2.pkl", "rb") as f:   
    cbf_model = pickle.load(f)

# Load XGBoost meta model
meta_model = xgb.Booster()
meta_model.load_model("../models/hybrid_meta_xgb.json")

# Waktu sekarang otomatis (WIB)
tz = pytz.timezone("Asia/Jakarta")
now = datetime.now(tz)
current_day = now.strftime("%A")  # "Monday", "Tuesday", dst
current_time = now.time()

# Map nama hari Indonesia ke English
day_map = {
    "Senin": "Monday",
    "Selasa": "Tuesday",
    "Rabu": "Wednesday",
    "Kamis": "Thursday",
    "Jumat": "Friday",
    "Sabtu": "Saturday",
    "Minggu": "Sunday"
}

# Simpan sementara interaksi user (ganti dengan DB sesungguhnya)
user_interactions = {}  # format: { user_id: { place_id: {action, rating, timestamp} } }
user_hybrid_models = {}  # { user_id: hybrid_model_instance }

# Load data
data, ratings_df = load_ratings()
_, places_df = load_places()
_, users_df = load_users()

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

# ===== Request Models =====
class InteractionItem(BaseModel):
    Place_Id: int
    Action: Literal["rate"]
    Rating: Optional[float] = None  # hanya jika action = rate

class SubmitInteractionsRequest(BaseModel):
    User_Id: int
    Interactions: List[InteractionItem] 

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def normalize_time_str(s):
    # Ubah 08.00 → 08:00 dan ganti en dash → dash
    s = s.replace(".", ":").replace("–", "-")
    # Hapus spasi ganda
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_open(working_hours_str, day, time_now):
    if pd.isna(working_hours_str):
        return False
    
    entries = working_hours_str.split(" | ")
    for entry in entries:
        if ":" not in entry:  # kalau formatnya aneh
            continue
        day_part, hours_part = entry.split(":", 1)
        day_part = day_part.strip()
        hours_part = normalize_time_str(hours_part.strip().lower())

        # Normalisasi hari
        day_part_en = day_map.get(day_part, day_part)
        if day_part_en.lower() == day.lower():
            if "buka 24 jam" in hours_part:
                return True
            if "tutup" in hours_part or "closed" in hours_part:
                return False
            if "-" in hours_part:
                open_str, close_str = hours_part.split("-")
                try:
                    open_time = datetime.strptime(open_str.strip(), "%H:%M").time()
                    close_time = datetime.strptime(close_str.strip(), "%H:%M").time()
                    return open_time <= time_now <= close_time
                except ValueError:
                    return False
    return False

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

# ====== CRUD Functions ======

def get_user_interactions(user_id: int):
    try:
        # 1. Sesuaikan nama kolom dengan database (huruf kecil semua)
        # 2. Tangani error dengan try...except
        resp = app.state.supabase.table("user_interactions").select("*").eq("user_id", user_id).execute()

        # Jika tidak ada data yang ditemukan, kembalikan dictionary kosong
        if not resp.data:
            return {}

        # 3. Sesuaikan nama kolom di bagian return dengan database (huruf kecil)
        return {
            row["place_id"]: {
                "action": row["action"],
                "rating": row["rating"],
                "timestamp": row["timestamp"]
            }
            for row in resp.data
        }
    
    except APIError as e:
        # Tangani error spesifik dari Supabase/PostgREST
        raise HTTPException(status_code=500, detail=f"Supabase API Error: {e.message}")
    except Exception as e:
        # Tangani error umum lainnya
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


def save_user_interactions(user_id: int, interactions: List[InteractionItem]):
    for inter in interactions:
        payload = {
            "user_id": user_id,
            "place_id": inter.Place_Id,
            "action": inter.Action,
            "rating": inter.Rating,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            resp = app.state.supabase.table("user_interactions").upsert(payload).execute()
            
            # Di versi terbaru, `resp.data` akan berisi data yang di-upsert jika berhasil.
            # Jadi, kita bisa cek keberadaan datanya.
            if not resp.data:
                raise HTTPException(
                    status_code=500, 
                    detail="Upsert operation failed for user_interactions with no data returned."
                )

        except APIError as e:
            # Tangani exception yang dilempar oleh Supabase client
            raise HTTPException(status_code=500, detail=f"Supabase API Error: {e.message}")
        except Exception as e:
            # Tangani kesalahan tak terduga lainnya
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def get_user_profile(user_id: int):
    try:
        # Gunakan 'user_id' yang sesuai dengan skema tabel
        resp = app.state.supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()

        # Jika tidak ada data, kembalikan None
        if not resp.data:
            return None
        
        # Jika ada data, kembalikan baris pertama
        # Kode ini akan mengembalikan dictionary seperti:
        # {'user_id': 123, 'lat': -6.123, 'long': 106.456, ...}
        return resp.data[0]

    except APIError as e:
        # Tangani error spesifik dari Supabase/PostgREST
        # Contohnya: kesalahan query, tabel tidak ditemukan, dll.
        raise HTTPException(status_code=500, detail=f"Supabase API Error: {e.message}")
    except Exception as e:
        # Tangani error umum lainnya
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


def save_user_profile(user_id: int, lat: float, lon: float):
    payload = {"user_id": user_id, "lat": lat, "long": lon} 
    try:
        # Coba jalankan pemanggilan Supabase
        resp = app.state.supabase.table("user_profiles").upsert(payload).execute()
        
        # Di versi terbaru, `resp.data` akan kosong jika upsert gagal
        if not resp.data:
            raise HTTPException(status_code=500, detail="Upsert operation failed with no data returned.")

    except APIError as e:
        # Tangkap exception yang dilempar oleh Supabase
        raise HTTPException(status_code=500, detail=f"Supabase API Error: {e.message}")
    except Exception as e:
        # Tangkap kesalahan lain yang tidak terduga
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Fit CBF user profiles (penting supaya rekomendasi bisa jalan)
cbf_model.fit(ratings_df)

# Instantiate hybrid model
hybrid_model = HybridRecommenderWithXGB(svd_model, cbf_model, meta_model)

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
    result = [{
        "Place_Id": int(row['Place_Id']),
        "Place_Name": row['Place_Name'],
        "Category": row['Category'],
        "City": row['City'],
        "Rating": None
    } for _, row in selected_places.iterrows()]
    if not result:
        raise HTTPException(status_code=404, detail="Tidak ada tempat ditemukan.")
    return result

# ========== POST rekomendasi dari input user baru ============
@app.post("/submit_preference")
def submit_user_preferences(data: PreferenceSubmission, max_distance_km: float = 10):
    user_id = data.User_Id
    save_user_profile(user_id, data.Latitude, data.Longitude)
    interactions = [InteractionItem(Place_Id=p.Place_Id, Action="rate", Rating=p.Rating) for p in data.Preferences]
    save_user_interactions(user_id, interactions)
    seen_places = set(p.Place_Id for p in data.Preferences)
    temp_df = pd.DataFrame({
        "User_Id": [user_id] * len(data.Preferences),
        "Place_Id": [p.Place_Id for p in data.Preferences],
        "Place_Ratings": [p.Rating for p in data.Preferences]
    })
    cbf_new = type(cbf_model)(places_df, temp_df, users_df)
    cbf_new.fit(temp_df)
    hybrid_for_new_user = HybridRecommenderWithXGB(svd_model, cbf_new, meta_model)
    user_hybrid_models[user_id] = hybrid_for_new_user
    recommendations = hybrid_for_new_user.recommend(user_id, seen_places, top_n=20)
    result = []
    for place_id, score in recommendations:
        place = places_df[places_df['Place_Id'] == place_id].iloc[0]
        open_status = is_open(place['Working_Hours'], current_day, current_time)
        distance = haversine(data.Latitude, data.Longitude, place['Lat'], place['Long'])
        result.append({
            "Place_Id": int(place_id),
            "Place_Name": place['Place_Name'],
            "Category": place['Category'],
            "City": place['City'],
            "Distance_km": round(distance, 2),
            "Score": float(round(score, 4)),
            "Is_Open": open_status
        })
    result.sort(key=lambda x: (not x["Is_Open"], x["Distance_km"], -x["Score"]))
    return {"User_Id": user_id, "Recommendations": result[:20]}

# ===== GET: Ambil tempat yang BELUM user rating =====
@app.get("/interactions")
def get_unrated_places(user_id: int = Query(...)):
    rated_places = {pid for pid, inter in get_user_interactions(user_id).items() if inter["action"] == "rate"}
    unique_places = places_df.drop_duplicates(subset=["Place_Id"])
    unrated_places = unique_places[~unique_places['Place_Id'].isin(rated_places)]
    return [{
        "Place_Id": int(row['Place_Id']),
        "Place_Name": row['Place_Name'],
        "Category": row['Category'],
        "City": row['City']
    } for _, row in unrated_places.iterrows()]

# ===== POST: Submit interaksi baru & update profil user =====
@app.post("/submit_interactions")
def submit_interactions(data: SubmitInteractionsRequest):
    save_user_interactions(data.User_Id, data.Interactions)
    user_data = get_user_interactions(data.User_Id)
    temp_data = {
        "User_Id": [],
        "Place_Id": [],
        "Place_Ratings": []
    }
    for pid, inter in user_data.items():
        if inter["action"] == "rate" and inter["rating"] is not None:
            temp_data["User_Id"].append(data.User_Id)
            temp_data["Place_Id"].append(pid)
            temp_data["Place_Ratings"].append(inter["rating"])
    temp_df = pd.DataFrame(temp_data)
    cbf_new = type(cbf_model)(places_df, temp_df, users_df)
    cbf_new.fit(temp_df)
    hybrid_for_new_user = HybridRecommenderWithXGB(svd_model, cbf_new, meta_model)
    user_hybrid_models[data.User_Id] = hybrid_for_new_user
    return {"status": "success", "message": "Interactions saved and user profile updated"}

# ===== GET: Ambil rekomendasi terbaru berdasarkan histori interaksi =====
@app.get("/recommendation")
def get_recommendations(user_id: int = Query(...), top_n: int = 20):
    user_data = get_user_interactions(user_id)
    seen_places = set(user_data.keys())
    profile = get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    model_to_use = user_hybrid_models.get(user_id, hybrid_model)
    # Tambahkan baris ini
    if model_to_use == hybrid_model:
        print(f"[DEBUG] Menggunakan model global (fallback) untuk User {user_id}")
    else:
        print(f"[DEBUG] Menggunakan model personalisasi untuk User {user_id}")
    recommendations = model_to_use.recommend(user_id, seen_places, top_n=top_n)
    result = []
    for place_id, score in recommendations:
        place = places_df[places_df['Place_Id'] == place_id].iloc[0]
        open_status = is_open(place['Working_Hours'], current_day, current_time)
        distance = haversine(profile["lat"], profile["long"], place['Lat'], place['Long'])
        result.append({
            "Place_Id": int(place_id),
            "Place_Name": place['Place_Name'],
            "Category": place['Category'],
            "City": place['City'],
            "Distance_km": round(distance, 2),
            "Score": float(round(score, 4)),
            "Is_Open": open_status
        })
    result.sort(key=lambda x: (not x["Is_Open"], x["Distance_km"], -x["Score"]))
    return {"User_Id": user_id, "Recommendations": result[:20]}

    

