import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

data = pd.read_csv('em_dataset.csv')
X = data[['Height']]

model = GaussianMixture(n_components=2)
model.fit(X)
labels = model.predict(X)

plt.scatter(X, [0]*len(X), c=labels, cmap='coolwarm')
plt.title("GMM Clustering via EM Algorithm")
plt.xlabel("Height")
plt.show()

print("Means of clusters:", model.means_.flatten())
print("Weights:", model.weights_)
