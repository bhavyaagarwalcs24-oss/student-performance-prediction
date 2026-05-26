import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("student_data.csv")

# Features
X = df[['Study_Hours',
        'Attendance',
        'Previous_Marks',
        'Assignment_Score']]

# Target
y = df['Final_Marks']

# Train model
model = RandomForestRegressor()

model.fit(X, y)

# Title
st.title("Student Performance Prediction")

# Inputs
hours = st.number_input("Study Hours")

attendance = st.number_input("Attendance")

previous = st.number_input("Previous Marks")

assignment = st.number_input("Assignment Score")

# Button
if st.button("Predict Marks"):

    new_data = pd.DataFrame([[hours,
                              attendance,
                              previous,
                              assignment]],
                            columns=['Study_Hours',
                                     'Attendance',
                                     'Previous_Marks',
                                     'Assignment_Score'])

    result = model.predict(new_data)

    st.success(f"Predicted Final Marks: {result[0]:.2f}")