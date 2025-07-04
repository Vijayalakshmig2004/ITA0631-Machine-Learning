import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Sample dummy data
data = {
    'Age': [25, 40, 35, 28, 50],
    'Income': [30000, 50000, 40000, 32000, 60000],
    'Credit_Score': ['Good', 'Poor', 'Average', 'Good', 'Poor'],
    'Loan_Approved': ['Yes', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

le = LabelEncoder()
df['Credit_Score'] = le.fit_transform(df['Credit_Score'])
df['Loan_Approved'] = le.fit_transform(df['Loan_Approved'])

X = df[['Age', 'Income', 'Credit_Score']]
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
