# Importar Librerias
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Importar modulos de sklearn
# descomposition tiene todo lo relacionado
# con reducción de la dimensionalidad
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv('./data/heart.csv')

    print(df_heart.head(5))

    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    kpca = KernelPCA(n_components=4, kernel='rbf') # Linear - Poly
    kpca.fit(X_train)

    df_train = kpca.transform(X_train)
    df_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')

    logistic.fit(df_train, y_train)
    print(f"SCORE KPCA:  {logistic.score(df_test, y_test)}")
