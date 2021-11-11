import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':
    dataset = pd.read_csv('./data/happiness.csv')
    print(dataset.head(5))

    X = dataset.drop(['country','rank','score'], axis=1)
    y = dataset[['score']]

    reg = RandomForestRegressor()

    parametros = {
        'n_estimators' : range(3,16),
        'criterion' : ['mse','mae'],
        'max_depth' : range(2,11)
    }

    rand_est = RandomizedSearchCV(reg, parametros, n_iter=30, cv = 3, scoring='neg_mean_absolute_error').fit(X, y)

    print("="*100)
    print(rand_est.best_estimator_)
    print("="*100)
    print(rand_est.best_params_)
    print("="*100)
    print("Valor Predecido: ",rand_est.predict(X.loc[[0]]))
    print("Valor Real: ", y.iloc[0])
