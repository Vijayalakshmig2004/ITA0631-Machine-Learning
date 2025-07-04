from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('id3_dataset.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

le = LabelEncoder()
X = X.apply(le.fit_transform)
y = le.fit_transform(y)

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

print(export_text(model, feature_names=list(data.columns[:-1])))

sample = [[2, 1, 0, 1]]  # Example: Overcast, Cool, Normal, Strong
print("Prediction:", model.predict(sample))
