import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Dummy monthly sales data
data = {
    'Month': [1, 2, 3, 4, 5, 6],
    'Sales': [1000, 1500, 2000, 2500, 3000, 3500]
}

df = pd.DataFrame(data)

X = df[['Month']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Predict future sales for month 7
future_month = np.array([[7]])
future_sales = model.predict(future_month)
print(f"Predicted Sales for Month 7: {future_sales[0]:.2f}")
