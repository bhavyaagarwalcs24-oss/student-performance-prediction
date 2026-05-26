import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# Load dataset
df = pd.read_csv("student_data.csv")

print(df.head())

# Features
X = df[['Study_Hours',
        'Attendance',
        'Previous_Marks',
        'Assignment_Score']]

# Target
y = df['Final_Marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestRegressor()

# Train model
model.fit(X_train, y_train)

# Predict test data
prediction = model.predict(X_test)

print("Predicted Marks:")
print(prediction)

# User input
hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))
previous = float(input("Enter Previous Marks: "))
assignment = float(input("Enter Assignment Score: "))

# New data
new_data = pd.DataFrame([[hours, attendance, previous, assignment]],
                        columns=['Study_Hours',
                                 'Attendance',
                                 'Previous_Marks',
                                 'Assignment_Score'])

# Final prediction
result = model.predict(new_data)

print("Predicted Final Marks:", result[0])
error = mean_absolute_error(y_test, prediction)

print("Model Error:", error)
plt.scatter(y_test, prediction)

plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")

plt.title("Actual vs Predicted")

plt.show()