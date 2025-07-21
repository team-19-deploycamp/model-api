import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, places_df, ratings_df):
        self.places_df = places_df
        self.ratings_df = ratings_df
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.user_profiles = {}

    def build_similarity_matrix(self):
        tfidf = TfidfVectorizer(stop_words='english')
        self.places_df['Content'] = self.places_df['Description'] + ' ' + self.places_df['Category'] + ' ' + self.places_df['City']
        self.tfidf_matrix = tfidf.fit_transform(self.places_df['Content'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def build_user_profile(self, user_id):
        user_ratings = self.ratings_df[self.ratings_df['User_Id'] == user_id]
        liked_places = user_ratings[user_ratings['Place_Ratings'] >= 4.0]
        if liked_places.empty:
            return np.zeros(self.tfidf_matrix.shape[1])
        liked_indices = [self.places_df[self.places_df['Place_Id'] == pid].index[0] for pid in liked_places['Place_Id']]
        liked_tfidf = self.tfidf_matrix[liked_indices]
        return np.asarray(liked_tfidf.mean(axis=0)).flatten()

    def fit(self):
        self.build_similarity_matrix()
        for user_id in self.ratings_df['User_Id'].unique():
            self.user_profiles[user_id] = self.build_user_profile(user_id)

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_profiles:
            return []
        user_profile = self.user_profiles[user_id]
        scores = cosine_similarity(user_profile.reshape(1, -1), self.tfidf_matrix).flatten()
        already_rated = set(self.ratings_df[self.ratings_df['User_Id'] == user_id]['Place_Id'])
        recommendations = []
        for idx in scores.argsort()[::-1]:
            place_id = self.places_df.iloc[idx]['Place_Id']
            if place_id not in already_rated:
                recommendations.append((place_id, scores[idx]))
            if len(recommendations) >= top_n:
                break
        return recommendations
