

import streamlit as st
import pandas as pd
import numpy as np
import joblib  
from joblib import load
from sklearn.preprocessing import OneHotEncoder



# LOADING THE MODELS

# Loading the roof PSF Limit model:

roof_psf_model = load('new_roof_model.joblib')

# Loading the analyzed attachment count model

count_model = load('get_analyzed_count_model.joblib')

# Loading the Ballast block quantity model:
ballast_model = load('working_model_ballast.joblib')



##########   LOADING THE ENCODERS 

encoder_roof = joblib.load('roof_encoder.pkl')


#############   LOADING THE SCALER 
scaler = joblib.load('scaler_work.save')


st.title("Prediction System")

Product_line_options = ['Eco-Top-HD', 'Eco-Top']
selected_Product_line_option = st.selectbox('Product line:', Product_line_options)

Reported_ASCE_options = ['ASCE 7-10', 'ASCE 7-16']
selected_Reported_ASCE_option = st.selectbox('Reported ASCE:', Reported_ASCE_options)

Analyzed_ASCE_options = ['ASCE 7-10', 'ASCE 7-16', 'IBC 2021 (ASCE 7-16)', 'IBC 2018 (ASCE 7-16)', 'IBC 2015 (ASCE 7-10)']
selected_Analyzed_ASCE_option = st.selectbox('Analyzed ASCE:', Analyzed_ASCE_options)

Exposure_Category_options = ['B', 'C', 'D']
selected_Exposure_Category_option = st.selectbox('Exposure Category:', Exposure_Category_options)

selected_Tilt_option = st.number_input("Tilt", format="%d", step=1)
selected_Analyzed_Windspeed_option = st.number_input("Analyzed Windspeed", format="%.6f")
selected_Analyzed_Snow_Load_option = st.number_input('Analyzed Snow Load:', format="%.6f")
selected_Ss_option = st.number_input('Ss:', format="%.6f")
selected_S1_option = st.number_input('S1:', format="%.6f")
selected_Roof_Deck_Height_option = st.number_input('Roof Deck Height (ft):', format="%.6f")
selected_Analyzed_Uplift_Capacity_option = st.number_input('Analyzed Uplift Capacity (lbs):', format="%d", step=1)
selected_Analyzed_Shear_Capacity_option = st.number_input('Analyzed Shear Capacity:', format="%d", step=1)
selected_Total_Project_Wattage_option = st.number_input('Total Project Wattage:', format="%d", step=1)
selected_Total_No_of_Panels_option = st.number_input('Total No. of Panels:', format="%d", step=1)
selected_Shade_Spacing_option = st.number_input('Shade Spacing (in):', format="%.6f")
selected_Module_Weight_option = st.number_input('Module Weight (lbs):', format="%.6f")
selected_Module_Length_option = st.number_input('Module Length (mm):', format="%d", step=1)
selected_Module_Width_option = st.number_input('Module Width (mm):', format="%d", step=1)
# selected_Attachments_per_100_Modules_option = st.number_input('Attachments per 100 Modules:', format="%.6f")
selected_Ballast_Weight_option = st.number_input('Ballast Weight (lbs):', format="%.6f")
# selected_Ballast_Block_Quantity_option = st.number_input('Ballast Block Quantity:', format="%d", step=1)
# selected_Ballast_Attach_Check_option = st.number_input('Ballast/Attach Check:', format="%d", step=1)

user_input = {
    'Tilt': selected_Tilt_option,
    'Analyzed Windspeed (mph)': selected_Analyzed_Windspeed_option,
    'Analyzed Snow Load': selected_Analyzed_Snow_Load_option,
    'Ss': selected_Ss_option,
    'S1': selected_S1_option,
    'Roof Deck Height (ft)': selected_Roof_Deck_Height_option,
    'Analyzed Uplift Capacity (lbs)': selected_Analyzed_Uplift_Capacity_option,
    'Analyzed Shear Capacity': selected_Analyzed_Shear_Capacity_option,
    'Total Project Wattage': selected_Total_Project_Wattage_option,
    'Total No. of Panels': selected_Total_No_of_Panels_option,
    'Shade Spacing (in)': selected_Shade_Spacing_option,
    'Module Weight (lbs)': selected_Module_Weight_option,
    'Module Length (mm)': selected_Module_Length_option,
    'Module Width (mm)': selected_Module_Width_option,
    # 'Attachments per 100 Modules': selected_Attachments_per_100_Modules_option,
    'Ballast Weight (lbs)': selected_Ballast_Weight_option,
    # 'Ballast Block Quantity': selected_Ballast_Block_Quantity_option,
    # 'Ballast/Attach Check': selected_Ballast_Attach_Check_option,
    'Product Line': selected_Product_line_option,
    'Reported ASCE': selected_Reported_ASCE_option,
    'Analyzed ASCE': selected_Analyzed_ASCE_option,
    'Exposure Category': selected_Exposure_Category_option
}

col1, col2, col3 = st.columns([2,1,2])

col2.markdown(
    """
    <style>
    div.stButton > button {
        width: 200px !important; /* Adjust the width as needed */
        margin-left: -8vh;
    }
    </style>
    """,
    unsafe_allow_html=True
)


if col2.button('Predict',  type="secondary"):
    input_df = pd.DataFrame([user_input])
    categorical_features = ['Product Line', 'Reported ASCE', 'Analyzed ASCE', 'Exposure Category']
    
    encoded_data = encoder_roof.transform(input_df[categorical_features])
    encoded_feature_names = encoder_roof.get_feature_names_out(categorical_features)


     

    
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names)



    
    input_df = input_df.drop(categorical_features, axis=1)
    final_df = pd.concat([input_df, encoded_df], axis=1)

   
    
    roof_prediction = roof_psf_model.predict(final_df)
    st.write(f"Predicted Roof PSF Limit: {roof_prediction[0]}")
    print(f"Predicted Roof PSF Limit: {roof_prediction[0]}")
    


    count_prediction = count_model.predict(final_df)
    st.write(f"Predicted Analyzed Attachment Count: {count_prediction[0]}")
    print(f"Predicted Analyzed Attachment Count: {count_prediction[0]}")


    # Scale the user input
    user_input_scaled = scaler.transform(final_df)

    ballast_prediction = ballast_model.predict(user_input_scaled)
    st.write(f"Predicted Ballast Block Quantity: {ballast_prediction[0]}")
    print(f"Predicted Ballast Block Quantity: {ballast_prediction[0]}")
