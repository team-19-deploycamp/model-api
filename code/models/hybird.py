import xgboost as xgb

class HybridRecommenderWithXGB:
    def __init__(self, svd_model, cbf_model):
        self.svd = svd_model
        self.cbf = cbf_model
        self.meta_model = None

    def fit_meta(self, trainset):
        X, y = [], []
        for uid_inner, iid_inner, true_r in trainset.all_ratings():
            uid = trainset.to_raw_uid(uid_inner)
            iid = trainset.to_raw_iid(iid_inner)
            est_svd = self.svd.predict(uid, iid).est
            est_cbf = self.cbf.predict_single(uid, iid)
            X.append([est_svd, est_cbf])
            y.append(true_r)

        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'seed': 42,
        }
        self.meta_model = xgb.train(params, dtrain, num_boost_round=100)


    def predict(self, uid, iid):
        est_svd = self.svd.predict(uid, iid).est
        est_cbf = self.cbf.predict_single(uid, iid)
        dtest = xgb.DMatrix([[est_svd, est_cbf]])
        pred = self.meta_model.predict(dtest)[0]
        return pred

    def test(self, testset):
        predictions = []
        for uid, iid, true_r in testset:
            try:
                est = self.predict(uid, iid)
            except Exception:
                est = 3.0
            predictions.append((uid, iid, true_r, est, None))
        return predictions