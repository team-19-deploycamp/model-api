from surprise import SVD

def SVDModel():
    """Initialize and return the collaborative filtering model."""
    return SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
