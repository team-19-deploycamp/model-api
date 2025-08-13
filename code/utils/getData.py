import pandas as pd
import os 
from surprise import Dataset, Reader

# Tentukan folder root project (deploycamp-capstone)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

def load_ratings(path=None):
    if path is None:
        path = os.path.join(DATASET_DIR, 'ratings.csv')
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(df.Place_Ratings.min(), df.Place_Ratings.max()))
    data = Dataset.load_from_df(df[['User_Id', 'Place_Id', 'Place_Ratings']], reader)
    return data, df

def load_places(path=None):
    if path is None:
        path = os.path.join(DATASET_DIR, 'places_v2.csv')
    df = pd.read_csv(path)
    df['Description'] = df['Description'].fillna('')
    df['Category'] = df['Category'].fillna('')
    df['City'] = df['City'].fillna('')
    df['Price'] = df['Price'].fillna(df['Price'].mean())
    return None, df

def load_users(path=None):
    if path is None:
        path = os.path.join(DATASET_DIR, 'users.csv')
    df = pd.read_csv(path)
    df['Location'] = df['Location'].fillna('Unknown')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return None, df
