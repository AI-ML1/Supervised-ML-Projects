import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model
model = joblib.load("/Users/hassi/Desktop/Projects/Project_2_dep/LinearRegression.pkl")


# Load the model accuracy
with open("/Users/hassi/Desktop/Projects/Project_2_dep/accuracy.txt", "r") as f:
    accuracy = f.read()

# Feature names
feature_names = [
    'levy', 'manufacturer', 'model', 'prod_year', 'engine_volume',
    'mileage', 'category', 'fuel_type', 'cylinders', 'color', 'airbags',
    'is_turbo', 'leather_interior_yes', 'gear_box_type_manual',
    'gear_box_type_tiptronic', 'gear_box_type_variator',
    'drive_wheels_front', 'drive_wheels_rear', 'wheel_right_hand_drive'
]

# Set up the Streamlit interface
st.title("Car Price Prediction with Linear Regression")
st.write(f"Model Accuracy: **{accuracy} %**")

# Custom CSS for styling
st.markdown("""
<style>
 /* Set background color */
 body {
     background-color: #F5F5F5;
 }
 /* Customize title */
 .title {
     font-size: 30px;
     color: #117A65;
     text-align: center;
     font-weight: bold;
 }
 /* Customize subtitle */
 .subtitle {
     font-size: 20px;
     color: #117A65;
     text-align: center;
     font-style: italic; 
 }
 /* Customize sidebar */
 .sidebar-title {
     font-size: 24px;
     color: red;
     text-align: center;
     font-weight: bold;    
 }
 /* Center align content */
 .main-content {
     max-width: 80%;
     margin: auto;
     padding: 20px;
     background-color: #FFFFFF;
     border-radius: 10px;
     box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
 }
</style>
""", unsafe_allow_html=True)

# Title and Subtitle with custom style
st.markdown("<p class='title'> Linear Regression Model for Car Price Prediction </p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'> Enter Details below to get a prediction </p>", unsafe_allow_html=True)

# Sidebar with Accuracy
st.sidebar.markdown("<p class='sidebar-title'> Model Performance </p>", unsafe_allow_html=True)
st.sidebar.write(f"### Model Accuracy: **{accuracy}%**")

# Main content box
st.markdown("<div class='main-content'>", unsafe_allow_html=True)
st.write("### Please enter the required details below and click 'Predict'")

# Input fields for user to enter values
input_data = []
for feature in feature_names:
    if feature in ['levy', 'manufacturer', 'model', 'prod_year', 'engine_volume','mileage']:
        value = st.number_input(f"Enter value for {feature}. Note: -1.0 <= data <= 1.0")
    elif feature in ['is_turbo', 'leather_interior_yes', 'gear_box_type_manual', 'gear_box_type_tiptronic', 
                     'gear_box_type_variator', 'drive_wheels_front', 'drive_wheels_rear', 'wheel_right_hand_drive']:
        value = st.selectbox(f"Enter value for {feature}.", ['1', '0'])
    elif feature in ['category']:
        value = st.number_input(f"Enter value for {feature}. Note: 0<=data<=9", min_value=0, max_value=10, step=1)
    elif feature in ['fuel_type']:
        value = st.number_input(f"Enter value for {feature}. Note: 0<=data<=5", min_value=0, max_value=6, step=1)
    elif feature in ['cylinders', 'color', 'airbags']:
        value = st.number_input(f"Enter value for {feature}.", min_value=0, max_value=20, step=1)
    else:
        value = st.number_input(f"Enter value for {feature}.", min_value=0, max_value=100, step=1)
    input_data.append(value)

# Predict when user clicks the button
if st.button("Predict"):
    try:
        # Convert input data to a numpy array
        input_array = np.array([input_data]).astype(float)

        # Prediction
        prediction = model.predict(input_array)[0]

        # Display the predicted car price
        st.write(f"### Predicted Car Price: ${prediction:.2f}")
    except Exception as e:
        # Handle any exceptions that may occur
        st.write(f"Error: {e}")

# Button for streamlit to check if the app is running correctly
st.title("🚀 Streamlit is Running!")
st.write("If you see this message, Streamlit is working correctly.")

# Optional: Feedback on button click
if st.button("Click Me"):
    st.write("✅ Button Clicked!")
    
# Data preprocessing function for model improvements (useful if you load more data for retraining)
def preprocess_data(input_data):
    # Convert input data into a pandas DataFrame for preprocessing
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Handle missing values by filling with 0 (or other strategies like mean, median, etc.)
    input_df = input_df.apply(pd.to_numeric, errors='coerce')
    input_df = input_df.fillna(0)
    
    # Handle categorical variables (e.g., one-hot encoding)
    categorical_features = ["manufacturer", "model", "category", "fuel_type", "color"]
    input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)
    
    return input_df