import pandas as pd
from surprise import Dataset, Reader

def load_ratings(path='../../dataset/tourism_rating.csv'):
    df = pd.read_csv(path)
    reader = Reader(rating_scale=(df.Place_Ratings.min(), df.Place_Ratings.max()))
    data = Dataset.load_from_df(df[['User_Id', 'Place_Id', 'Place_Ratings']], reader)
    return data, df
