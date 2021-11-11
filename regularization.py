# Cargar librerias
import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    dataset = pd.read_csv('./data/happiness.csv')
    print(dataset.describe())
    print("\n")

    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    print(X.shape, y.shape)
    print("\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    # Entre mayor sea el Alpha mas penalización van a tener los features
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=0.02).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    modelElasticNet = ElasticNet(alpha=0.02,random_state=42).fit(X_train, y_train)
    y_precit_elasticnet = modelElasticNet.predict(X_test)

    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print(f"Linear Loss: {linear_loss}")

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print(f"Lasso Loss: {lasso_loss}")

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print(f"Ridge Loss: {ridge_loss}")

    elasticnet_loss = mean_squared_error(y_test, y_precit_elasticnet)
    print(f"ElasticNet Loss: {elasticnet_loss}")
    print("\n")

    print("="*64)
    print("Coef LASSO: ")
    print(modelLasso.coef_)
    print("\n")

    print("="*64)
    print("Coef Ridge: ")
    print(modelRidge.coef_)
    print("\n")

    print("="*64)
    print("Coef ElasticNet: ")
    print(modelElasticNet.coef_)
    print("\n")
