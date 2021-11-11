import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import MeanShift

if __name__ == '__main__':
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(5))

    X = dataset.drop('competitorname',axis=1)

    meanshift = MeanShift().fit(X) # bandwith: parametro importante, lo elige por defecto con una tecnica matematica

    print(f"Numero de etiquetas: {max(meanshift.labels_)}")

    print('='*160)

    print(meanshift.cluster_centers_)

    dataset[['group']] = meanshift.labels_

    print('='*160)

    print(dataset.head(10))

    sns.scatterplot(data=dataset, x="sugarpercent", y="winpercent", hue="group", palette="deep")
    plt.grid()
    plt.show()
