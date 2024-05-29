from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from lightgbm.callback import early_stopping
from joblib import dump
import optuna
from optuna.samplers import TPESampler
import gc
import os


class Base_lgbm_model(ABC):
    def __init__(self, X, y, base_params={}, model_name="base_lgbm_model"):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.base_params = base_params

    def r2_metric(self, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        r2 = r2_score(y_true, y_pred)
        return "r2", r2, True

    @abstractmethod
    def objective(self, trial):
        raise NotImplementedError

    def fit(self, params):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(self.X)):
            X_train = self.X.iloc[train_idx]
            X_valid = self.X.iloc[valid_idx]

            y_train = self.y.iloc[train_idx]
            y_valid = self.y.iloc[valid_idx]

            model = LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric=self.r2_metric,
                callbacks=[early_stopping(250, verbose=False)],
            )

            valid_preds = model.predict(X_valid)
            score = r2_score(y_valid, valid_preds)
            scores.append(score)

            del model
            gc.collect()

        return scores

    def optimize(self, n_trials=100):
        study = optuna.create_study(direction="maximize", sampler=TPESampler())
        study.optimize(
            self.objective, n_trials=n_trials, show_progress_bar=True, n_jobs=5
        )

        print("Best trial:")
        trial = study.best_trial
        print(" Value: ", trial.value)
        print(" Params: ")
        for key, value in trial.params.items():
            print(f" {key}: {value}")

        if not os.path.exists("models"):
            os.mkdir("models")

        params_dict = trial.params
        params_dict.update(self.base_params)

        dump(params_dict, f"models/{self.model_name}.joblib")
        return study.best_params
