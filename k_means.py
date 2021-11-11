import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    dataset = pd.read_csv('./data/candy.csv')
    print(dataset.head(5))

X = dataset.drop(['competitorname'], axis=1)

kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)

print("Total de centros: ", len(kmeans.cluster_centers_))
print('='*64)

predictions = kmeans.predict(X)
print(predictions)

dataset[['group']] = predictions
sns.scatterplot(data=dataset, x="sugarpercent", y="winpercent", hue="group", palette="deep")
plt.grid()
plt.show()
