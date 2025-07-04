import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dummy data (create your own or replace with real dataset)
data = {
    'RAM': [4, 6, 8, 3, 12],
    'ROM': [64, 128, 256, 32, 512],
    'Battery': [4000, 5000, 4500, 3000, 6000],
    'Price': [15000, 20000, 25000, 10000, 40000]
}

df = pd.DataFrame(data)

X = df[['RAM', 'ROM', 'Battery']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

print("Predicted Price for 6GB RAM, 128GB ROM, 5000mAh battery:")
print(model.predict([[6, 128, 5000]]))
