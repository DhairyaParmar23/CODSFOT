import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("Iris Flower Classification Started")


data = pd.read_csv("iris.csv")


print(data.head())


X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LogisticRegression()
model.fit(X_train, y_train)


print("\nEnter flower details:")
sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))


prediction = model.predict([[sl, sw, pl, pw]])

print("Predicted Species:", prediction[0])
