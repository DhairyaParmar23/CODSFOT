import os
print(os.listdir())
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Movie Rating Prediction Started")


data = pd.read_csv("movie.csv.csv", encoding='latin1')


print(data.head())


data = data.dropna()


data['Genre'] = data['Genre'].astype('category').cat.codes


data['Duration'] = data['Duration'].astype(str)
data['Duration'] = data['Duration'].str.replace(' min', '', regex=False)
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')


data['Votes'] = data['Votes'].astype(str)
data['Votes'] = data['Votes'].str.replace(',', '', regex=False)
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')


data = data.dropna()


X = data[['Genre', 'Votes', 'Duration']]
y = data['Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = LinearRegression()
model.fit(X_train, y_train)


print("\nEnter movie details:")
genre = int(input("Genre (number): "))
votes = float(input("Votes: "))
duration = float(input("Duration (minutes): "))


prediction = model.predict([[genre, votes, duration]])

print("Predicted Rating:", prediction[0])
