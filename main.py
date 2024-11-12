import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('/Users/interlink/Desktop/Deploy/trained_model.sav', 'rb'))

def breast_cancer_prediction(input_data):
    # Convert input data to float and handle missing values
    input_data = [0.0 if val == '' else float(val) for val in input_data]
    
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape numpy array as we are using one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction == 0:
        return "The breast cancer is Malignant"
    else:
        return "The breast cancer is Benign"

def main():
    st.title("Breast cancer prediction")
    
    # Collect input data from the user
    input_features = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", 
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", 
        "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", 
        "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", 
        "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", 
        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", 
        "concavity_worst", "concave_points_worst", "symmetry_worst", 
        "fractal_dimension_worst"
    ]
    
    user_input = [st.text_input(f"{feature.capitalize()}:") for feature in input_features]

    diagnosis = ""
    
    if st.button("Test"):
        diagnosis = breast_cancer_prediction(user_input)
        
    st.success(diagnosis)

if __name__=='__main__':
    main()
