from lgbm.base_model import Base_lgbm_model
import numpy as np

class GOSS_lgbm_model(Base_lgbm_model):
    def __init__(self, X, y, model_name = 'goss_lgbm_model'):
        super().__init__(X, y, model_name)

    def objective(self, trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
            "boosting_type": 'goss',
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 1),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 1),
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 1),
            "max_bin": trial.suggest_int("max_bin", 10, 1000),
            "random_state": 42,
            "n_jobs": 5,
            "metric": "rmse",
            "top_rate": trial.suggest_float("top_rate", 0.1, 0.5),
            "other_rate": trial.suggest_float("other_rate", 0.1, 0.5),
            "max_delta_step": trial.suggest_float("max_delta_step", 0, 1),
            "verbose": -1
        }
        scores = self.fit(params)
        return np.mean(scores)
    
