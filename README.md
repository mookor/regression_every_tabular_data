## A set of tools for selecting boosting parameters
1. Currently implemented only for regression tasks
2. Currently implemented only lightGBM and Catboost

| Task type  |        Models|        ModelType  |     status |
| ------------- | ------------- | ------------- |------------- |
| Regression    |     lightGBM  |           gbdt| :white_check_mark: | 
| Regression    |     lightGBM  |           goss| :white_check_mark: |
| Regression    |     lightGBM  |             rf| :white_check_mark: |
| Regression    |     Catboost  |      Depthwise| :white_check_mark: |
| Regression    |     Catboost  |      Lossguide| :white_check_mark: |
| Regression    |     Catboost  |  SymmetricTree| :white_check_mark: |
| Classification   |     -  |  - | :negative_squared_cross_mark:  - now in work|


## Example

import libraries:
```
from lightgbm import LGBMRegressor
from lgbm.gbdt_lgbm_model import GBDT_lgbm_model
from sklearn.metrics import mean_squared_error
import pandas as pd
```

define dataset:
```
X = pd.DataFrame({"x1": [1, 2, 3, 4, 5] * 100})
y = pd.Series([0.1 , 0.2, 0.3, 0.4, 0.5] * 100)
```

define model to optimize:
```
model_name = "X*0.1_model"  # it will be saved in models folder by path './models/{model_name}.joblib'
model = GBDT_lgbm_model(X, y, model_name=model_name)  # model, with gbdt boosting type
```

run optimize cycle:
```
model.optimize(10)
```

load optimized params:
```
optimized_model_params = load(f"models/{model_name}.joblib")  # load optimized model parameters
optimized_model_params['verbose'] = -1                        # turn off verbose
```

define model with optimized parameters:
```
optimized_model = LGBMRegressor(**optimized_model_params)
```

fit / predict:
```
optimized_model.fit(X, y)                                     # fit model with optimized parameters
model_prediction = optimized_model.predict(X)                 # predict
```

check results:
```
print("Mean squared error: %.4f" % mean_squared_error(y, model_prediction))
```