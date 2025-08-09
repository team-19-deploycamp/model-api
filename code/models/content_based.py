import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, places_df, ratings_df, users_df):
        self.places_df = places_df.copy()
        self.ratings_df = ratings_df
        self.users_df = users_df
        self.vectorizer = TfidfVectorizer()
        self.embeddings = None
        self.similarity_matrix = None
        self.user_profiles = {}

    def build_similarity_matrix(self):
        self.places_df['Content'] = (
            self.places_df['Description'].fillna('') + ' ' +
            self.places_df['Category'].fillna('') + ' ' +
            self.places_df['City'].fillna('')
        )

        self.embeddings = self.vectorizer.fit_transform(self.places_df['Content'])
        self.similarity_matrix = cosine_similarity(self.embeddings)

    def get_age_segment_keywords(self, age):
        if age < 25:
            return "petualangan anak muda hemat"
        elif age <= 40:
            return "modern trendi nyaman"
        else:
            return "tenang budaya warisan"

    def build_user_profile(self, user_id, ratings_subset):
        user_ratings = ratings_subset[ratings_subset['User_Id'] == user_id]
        liked_places = user_ratings[user_ratings['Place_Ratings'] >= 4.0]

        if liked_places.empty:
            return np.zeros(self.embeddings.shape[1])

        profile = np.zeros(self.embeddings.shape[1])
        total_weight = 0.0

        for _, row in liked_places.iterrows():
            place_id = row['Place_Id']
            rating = row['Place_Ratings']
            idx = self.places_df[self.places_df['Place_Id'] == place_id].index
            if len(idx) == 0:
                continue
            embedding = self.embeddings[idx[0]].toarray().flatten()
            profile += embedding * rating
            total_weight += rating

        if total_weight == 0:
            return np.zeros(self.embeddings.shape[1])

        user_profile = profile / total_weight

        # Tambahkan pengaruh umur
        try:
            age = self.users_df[self.users_df['User_Id'] == user_id]['Age'].values[0]
            age_keywords = self.get_age_segment_keywords(age)
            age_vector = self.vectorizer.transform([age_keywords]).toarray().flatten()
            user_profile = 0.7 * user_profile + 0.3 * age_vector
        except Exception:
            pass

        return user_profile

    def fit(self, train_ratings_df):
        self.build_similarity_matrix()
        for user_id in train_ratings_df['User_Id'].unique():
            self.user_profiles[user_id] = self.build_user_profile(user_id, train_ratings_df)

    def recommend(self, user_id, seen_places, top_n=10):
        if user_id not in self.user_profiles:
            return []

        user_profile = self.user_profiles[user_id]
        scores = cosine_similarity([user_profile], self.embeddings).flatten()

        recommendations = []
        for idx in scores.argsort()[::-1]:
            place_id = self.places_df.iloc[idx]['Place_Id']
            if place_id not in seen_places:
                recommendations.append((place_id, scores[idx]))
            if len(recommendations) >= top_n:
                break
        return recommendations
