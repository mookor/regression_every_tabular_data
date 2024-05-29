from joblib import load
from lgbm.gbdt_lgbm_model import GBDT_lgbm_model
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd

# define dataset
X = pd.DataFrame({"x1": [1, 2, 3, 4, 5] * 100})
y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5] * 100)

# define model
model_name = "X*0.1_model"  # it will be saved in models folder by path './models/{model_name}.joblib'

model = GBDT_lgbm_model(X, y, model_name=model_name)  # model, with gbdt boosting type
model.optimize(10)  # 10 trials

optimized_model_params = load(
    f"models/{model_name}.joblib"
)  # load optimized model parameters

optimized_model_params["verbose"] = -1  # turn off verbose

optimized_model = LGBMRegressor(
    **optimized_model_params
)  # define model with optimized parameters

optimized_model.fit(X, y)  # fit model with optimized parameters
model_prediction = optimized_model.predict(X)  # predict

print(
    "Mean squared error: %.4f" % mean_squared_error(y, model_prediction)
)  # print mean squared error
