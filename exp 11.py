import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Sample data
data = {
    'Income': [50000, 60000, 35000, 80000, 20000, 100000],
    'Age': [25, 35, 45, 32, 50, 28],
    'LoanAmount': [2000, 3000, 1500, 4000, 1000, 5000],
    'CreditScore': ['Good', 'Good', 'Average', 'Good', 'Poor', 'Good']
}

df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
df['CreditScore'] = le.fit_transform(df['CreditScore'])

X = df[['Income', 'Age', 'LoanAmount']]
y = df['CreditScore']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=le.classes_))
print("hi")
