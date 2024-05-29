"""
Train LGBM models for a "Regression with a Flood Prediction Dataset" 
https://www.kaggle.com/competitions/playground-series-s4e5
"""

from lgbm.gbdt_lgbm_model import GBDT_lgbm_model
from lgbm.rf_lgbm_model import RF_lgbm_model
from lgbm.goss_lgbm_model import GOSS_lgbm_model
from dataloader import DataLoader
import gc
import time

train_path = "data/train.csv"
test_path = "data/test.csv"

data_loader = DataLoader(train_path, test_path)
X, y = data_loader.get_train_set()
n_trials = 1

models = [RF_lgbm_model, GBDT_lgbm_model, GOSS_lgbm_model]
for model in models:
    model = model(X, y)
    print(f"--------------{model.model_name}--------------")
    start_time = time.time()
    model.optimize(n_trials)
    print(
        f"model: {model.model_name}, optimize time: {(time.time() - start_time) // 60} min ({(time.time() - start_time) % 60} sec)"
    )

    del model
    gc.collect()
