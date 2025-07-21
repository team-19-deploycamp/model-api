from surprise import KNNBasic

def KNN():
    sim_options = {'name': 'cosine', 'user_based': True}
    return KNNBasic(sim_options=sim_options)
