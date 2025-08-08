import streamlit as st
import pandas as pd
import numpy as np
# We no longer use joblib; instead we train the model at runtime
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset and train classifier
iris = load_iris()
X, y = iris.data, iris.target
clf = RandomForestClassifier()
clf.fit(X, y)

def predict(data: np.ndarray) -> np.ndarray:
    """Return a string label including the class index and name (e.g., '0-setosa')."""
    pred_idx = clf.predict(data)[0]
    return np.array([f"{pred_idx}-{iris.target_names[int(pred_idx)]}"])

def class_to_image(class_name: str) -> str:
    """Map class names to image filenames. Replace the filenames with actual images in your repository."""
    if class_name == "setosa":
        return "images/setosa.jpg"
    elif class_name == "versicolor":
        return "images/versicolor.jpg"
    elif class_name == "virginica":
        return "images/virginica.jpg"
    # default fallback
    return ""

# Streamlit app UI
st.title('Classifying Iris flowers')
st.markdown('Model to classify iris flowers into (setosa, versicolor, virginica) based on their sepal/petal and length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal length (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

st.text('')

if st.button("Predict type of Iris"):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    # Extract the class name after the hyphen
    predicted_class_name = result[0].split('-')[1]
    st.text(predicted_class_name)
    # Display corresponding image
    image_path = class_to_image(predicted_class_name)
    st.image(image_path, use_column_width=True)

st.text('')