import pandas as pd
from surprise import Dataset, Reader

def load_ratings(path='../../dataset/ratings.csv'):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(df.Place_Ratings.min(), df.Place_Ratings.max()))
    data = Dataset.load_from_df(df[['User_Id', 'Place_Id', 'Place_Ratings']], reader)
    return data, df

def load_places(path='../../dataset/places_v2.csv'):
    df = pd.read_csv(path)
    df['Description'] = df['Description'].fillna('')
    df['Category'] = df['Category'].fillna('')
    df['City'] = df['City'].fillna('')
    df['Price'] = df['Price'].fillna(df['Price'].mean())
    return None, df

def load_users(path='../../dataset/users.csv'):
    df = pd.read_csv(path)
    df['Location'] = df['Location'].fillna('Unknown')
    df['Age'] = df['Age'].fillna(df['Age'].median())
    return None, df
