import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

from sklearn.linear_model import (
    RANSACRegressor,
    HuberRegressor
)

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/happiness_corrupt.csv')
    print(dataset.head(5))

    X = dataset.drop(['country','score'], axis=1)
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    estimadores = {
        'SVR' : SVR(gamma='auto', C=1.0, epsilon=0.1), # Estimador
        'RANSAC' : RANSACRegressor(), # Meta Estimador, por defecto trabaja con un modelo de regresión lineal
        'HUBER' : HuberRegressor(epsilon=1.35) # Epsilon recomendado en la literatura para el 95% de los casos
    }

    for name,  estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print("="*64)
        print(f"{name} --> ", f"MSE: {mean_squared_error(y_test, predictions)}")
        plt.ylabel('Predicted Score')
        plt.xlabel('Real Score')
        plt.title(f'Predicted VS Real ({name})')
        plt.scatter(y_test, predictions)
        plt.plot(predictions, predictions,'r--')
        plt.grid()
        plt.show()
