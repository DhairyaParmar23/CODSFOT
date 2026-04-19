import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Program started")

data = pd.read_csv("titanic.csv")

print(data.head())

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

print("\nEnter details:")
pclass = int(input("Class (1/2/3): "))
sex = int(input("Gender (0=male, 1=female): "))
age = float(input("Age: "))
fare = float(input("Fare: "))

result = model.predict([[pclass, sex, age, fare]])

if result[0] == 1:
    print("Survived")
else:
    print("Did Not Survive") 



