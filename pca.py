# Importar Librerias
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

# Importar modulos de sklearn
# descomposition tiene todo lo relacionado
# con reducción de la dimensionalidad
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df_heart = pd.read_csv('./data/heart.csv')

    print(df_heart.head(5))
    print("-"*96)

    df_features = df_heart.drop(['target'], axis=1)
    df_target = df_heart['target']

    df_features = StandardScaler().fit_transform(df_features)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=42)

    # print(X_train.shape, y_train.shape)

    # El numero de componentes por defecto: n_components = min(n_muestras, n_features)
    n_components = 3
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    # batch_size (Pequeños bloques): No manda todos los datos a entrenar al mismo tiempo sino que manda pequeños bloques
    ipca = IncrementalPCA(n_components=n_components, batch_size=10)
    ipca.fit(X_train)

    print("***** Entrenando Modelol *****")
    logistic = LogisticRegression(solver='lbfgs')

    df_train = pca.transform(X_train)
    df_test = pca.transform(X_test)
    logistic.fit(df_train,y_train)
    print(f"SCORE PCA: {logistic.score(df_test, y_test)}")

    df_train = ipca.transform(X_train)
    df_test = ipca.transform(X_test)
    logistic.fit(df_train, y_train)
    print(f"SCORE IPCA: {logistic.score(df_test, y_test)}")

    plt.plot(range(len(pca.explained_variance_)),pca.explained_variance_ratio_,label="Explained Variance Ratio")
    plt.title("Explained Variance Ratio")
    plt.ylabel("percentage (%)")
    plt.xlabel("Explained Variance Ratio")
    plt.grid()
    plt.legend()
    plt.show()
