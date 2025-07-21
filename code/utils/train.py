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


def main():
    data, df = load_ratings()
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    models = {
        "SVD": SVDModel(),
        "KNNBasic": KNN(),
        "BaselineOnly": baselineOnly(),
        "NormalPredictor": normalPredictor()
    }

    for model_name, model in models.items():
        evaluate_model(model_name, model, trainset, testset)


if __name__ == "__main__":
    main()
