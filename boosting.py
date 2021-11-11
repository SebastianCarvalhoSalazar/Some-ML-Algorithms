import pandas as pd
import warnings
warnings.simplefilter("ignore")

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df_heart = pd.read_csv('./data/heart.csv')
    print(df_heart.describe())

    X = df_heart.drop(['target'],axis=1)
    y = df_heart[['target']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.35)

    boost = GradientBoostingClassifier(n_estimators=52).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print('='*64)
    print(accuracy_score(boost_pred, y_test))
