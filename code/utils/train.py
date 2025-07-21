import os
import sys
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.svd import SVDModel
from models.baseline import baselineOnly
from models.knn_basic import KNN
from models.normal_predictor import normalPredictor

from models.content_based import ContentBasedRecommender
from getData import load_ratings, load_places
from collections import defaultdict

from getData import load_ratings

def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = [], []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k)
        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0
        precisions.append(precision)
        recalls.append(recall)
    return sum(precisions)/len(precisions), sum(recalls)/len(recalls)

def evaluate_model(model_name, model, trainset, testset, n=10, k=10, threshold=4.0):
    print(f"\n=== Evaluating {model_name} ===")
    model.fit(trainset)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k=k, threshold=threshold)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Precision@{k}: {precision:.4f}")
    print(f"Recall@{k}: {recall:.4f}")

    top_n = get_top_n(predictions, n=n)
    sample_user = list(top_n.keys())[0]  # just pick first user for demo
    print(f"\nTop-{n} recommendations for User {sample_user}:")
    for iid, score in top_n[sample_user]:
        print(f"  Place ID {iid}, Predicted Rating: {score:.2f}")

def evaluate_cbf(cbf, ratings_df, top_n=10, threshold=4.0):
    precisions, recalls = [], []

    for user_id in ratings_df['User_Id'].unique():
        true_ratings = ratings_df[ratings_df['User_Id'] == user_id]
        true_liked = set(true_ratings[true_ratings['Place_Ratings'] >= threshold]['Place_Id'])

        recommended = cbf.recommend(user_id, top_n)
        recommended_ids = [pid for pid, _ in recommended]

        if not true_liked:
            continue

        n_rel = len(true_liked)
        n_rec_k = len(recommended_ids)
        n_rel_and_rec_k = len(set(recommended_ids) & true_liked)

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0

        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    print(f"\n=== Evaluating Content-Based Filtering ===")
    print(f"Precision@{top_n}: {avg_precision:.4f}")
    print(f"Recall@{top_n}: {avg_recall:.4f}")


def main():
    data, ratings_df = load_ratings()
    _, places_df = load_places()

    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    models = {
        "SVD": SVDModel(),
        "KNNBasic": KNN(),
        "BaselineOnly": baselineOnly(),
        "NormalPredictor": normalPredictor()
    }

    for model_name, model in models.items():
        evaluate_model(model_name, model, trainset, testset)

    cbf = ContentBasedRecommender(places_df, ratings_df)
    cbf.fit()
    evaluate_cbf(cbf, ratings_df)

        # Cetak tempat yang telah dirating user 1
    sample_user = 1
    user_ratings = ratings_df[ratings_df['User_Id'] == sample_user]
    user_ratings = user_ratings.sort_values(by='Place_Ratings', ascending=False)

    print(f"\nTempat yang dirating oleh User {sample_user}:")
    for _, row in user_ratings.iterrows():
        place_info = places_df[places_df['Place_Id'] == row['Place_Id']]
        if not place_info.empty:
            place_name = place_info.iloc[0].get('Place_Name', row['Place_Id'])  # fallback ke ID kalau tidak ada nama
        else:
            place_name = row['Place_Id']
        print(f"  {place_name} (ID: {row['Place_Id']}), Rating: {row['Place_Ratings']}")

    sample_user = 1
    print(f"\nTop-10 rekomendasi untuk User {sample_user} (Content-Based):")
    recommendations = cbf.recommend(sample_user, top_n=10)
    for place_id, score in recommendations:
        place = places_df[places_df['Place_Id'] == place_id].iloc[0]
        print(f"  {place['Place_Name']} (ID: {place_id}), Skor: {score:.4f}")


if __name__ == "__main__":
    main()
