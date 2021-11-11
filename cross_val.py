import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score,
    KFold
)

if __name__ == '__main__':
    dataset = pd.read_csv('./data/happiness.csv')

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset[['score']]

    model = DecisionTreeRegressor()
    score = cross_val_score(model, X, y, cv=6, scoring='neg_mean_squared_error')

    print("\n")
    print('='*128)
    print(np.abs(np.mean(score)))

    print('='*128)

    kf = KFold(n_splits=6, shuffle=True, random_state=42) # Pliegues, ¿Queremos que los datos se organicen aleatoriamente?, Semilla (Replicabilidad)
    for train, test in kf.split(dataset):
        print(train) # Lo que va a mandar al conjunto de train
        print(test)  # Lo que va a mandar al conjunto de test
        print('='*128)
