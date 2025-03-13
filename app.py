# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load Dataset
df = pd.read_csv('diabetes.csv')

# Ensure column names are standardized (title case)
df.columns = df.columns.str.strip()  # Remove spaces
df.columns = df.columns.str.replace(' ', '')  # Remove spaces inside column names

# HEADINGS
st.title('Diabetes Prediction App')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# FUNCTION TO TAKE USER INPUT
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    # Create a DataFrame with matching column names
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    return user_data

# Get user input
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Standardize user input to match model training
user_data_scaled = scaler.transform(user_data)

# MODEL TRAINING
rf = RandomForestClassifier()
rf.fit(X_train_scaled, y_train)

# PREDICTION
user_result = rf.predict(user_data_scaled)


# OUTPUT RESULTS
st.subheader('Your Report:')
st.title("You are Diabetic" if user_result[0] == 1 else "You are not Diabetic")

# Accuracy
st.subheader('Model Accuracy:')
accuracy = accuracy_score(y_test, rf.predict(X_test_scaled))
st.write(f"{accuracy * 100:.2f}%")

# VISUALIZATION
st.title('Visualized Patient Report')

# COLOR FUNCTION
color = 'red' if user_result[0] == 1 else 'blue'

# Graphs
def plot_graph(x_feature, y_feature, palette_color):
    fig = plt.figure()
    sns.scatterplot(x=df[x_feature], y=df[y_feature], hue=df['Outcome'], palette=palette_color)
    sns.scatterplot(x=user_data[x_feature], y=user_data[y_feature], s=150, color=color)
    plt.title(f'0 - Healthy & 1 - Diabetic ({x_feature} vs {y_feature})')
    st.pyplot(fig)

# Plot different health metrics
plot_graph('Age', 'Pregnancies', 'Greens')
plot_graph('Age', 'Glucose', 'magma')
plot_graph('Age', 'BloodPressure', 'Reds')
plot_graph('Age', 'SkinThickness', 'Blues')
plot_graph('Age', 'Insulin', 'rocket')
plot_graph('Age', 'BMI', 'rainbow')
plot_graph('Age', 'DiabetesPedigreeFunction', 'YlOrBr')

# # OUTPUT RESULTS
# st.subheader('Your Report:')
# st.title("You are Diabetic" if user_result[0] == 1 else "You are not Diabetic")

# # Accuracy
# st.subheader('Model Accuracy:')
# accuracy = accuracy_score(y_test, rf.predict(X_test_scaled))
# st.write(f"{accuracy * 100:.2f}%")
