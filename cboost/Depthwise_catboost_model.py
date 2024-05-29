from cboost.base_model import Base_catboost_model
import optuna
import numpy as np


class Depthwise_catboost_model(Base_catboost_model):
    def __init__(
        self,
        X,
        y,
        base_params={
            "random_seed": 42,
            "eval_metric": "R2",
            "boosting_type": "Plain",
            "grow_policy": "Depthwise",
        },
        model_name="depthwise_catboost_model",
    ):
        super().__init__(X, y, base_params, model_name)

    def objective(self, trial):
        boosting_type = "Plain"
        grow_policy = "Depthwise"
        all_score_functions = ["Cosine", "L2"]
        score_function = trial.suggest_categorical(
            "score_function", all_score_functions
        )

        params = {
            "task_type": "GPU",
            "devices": "0",
            "iterations": trial.suggest_int("iterations", 50, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 2, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
            "random_seed": 42,
            "boosting_type": boosting_type,
            "grow_policy": grow_policy,
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            "score_function": score_function,
            "feature_border_type": trial.suggest_categorical(
                "feature_border_type",
                ["GreedyLogSum", "MinEntropy", "Median", "UniformAndQuantiles"],
            ),
            "leaf_estimation_method": trial.suggest_categorical(
                "leaf_estimation_method", ["Gradient", "Newton"]
            ),
            "leaf_estimation_iterations": trial.suggest_int(
                "leaf_estimation_iterations", 1, 10
            ),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 0, 25),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "No", "MVS"]
            ),
            "eval_metric": "R2",
            "thread_count": -1,
        }

        if params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)

        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )

        scores = self.fit(params)
        return np.mean(scores)
