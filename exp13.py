import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dummy dataset
data = {
    'Year': [2010, 2012, 2014, 2016, 2018],
    'Kms_Driven': [30000, 40000, 25000, 35000, 20000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'Price': [3.5, 4.0, 3.0, 4.5, 5.0]
}

df = pd.DataFrame(data)
df = pd.get_dummies(df, columns=['Fuel_Type'], drop_first=True)

X = df[['Year', 'Kms_Driven', 'Fuel_Type_Petrol']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

print("Predicted price:", model.predict([[2020, 15000, 1]]))
