import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("dataset.csv")

X = data[["hours_studied", "attendance_percentage", "previous_score"]]
y = data["final_score"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)


sample_input = [[6, 80, 75]]
sample_output = model.predict(sample_input)
print("Predicted final score:", sample_output[0])
