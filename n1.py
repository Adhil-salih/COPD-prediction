import streamlit as st
import pickle 
from PIL import Image
import time
import base64


# Load model files 
model=pickle.load(open("COPD_Prediction.save","rb"))
scalar=pickle.load(open("COPD_scalar.save","rb"))
le1=pickle.load(open("COPD_Gender.save","rb"))
le2=pickle.load(open("COPD_Bloodpressure.save","rb"))
le3=pickle.load(open("COPD_Cholesterol.save","rb"))
le4=pickle.load(open("COPD_Smoking.save","rb"))
le5=pickle.load(open("COPD_Alcoholconsumption.save","rb"))
le6=pickle.load(open("COPD_Exercise.save","rb"))
le7=pickle.load(open("COPD_Familyhistory.save","rb"))

    
# Function to parse the image background securely
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    st.set_page_config(page_title="COPD Predictor", layout="centered")

    # BACKGROUND IMAGE INJECTION 
    try:
        bin_str = get_base64_image('image.png')
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bin_str}");
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
            }}

            /* Wraps your columns in semi-transparent card */
            [data-testid="stHorizontalBlock"] {{
                background-color: rgba(0, 0, 0, 0.75) !important;  
                padding: 30px !important;
                border-radius: 16px !important;
                box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.5) !important;
            }}
            
            /* Keep labels inside the card */
            label, p, .stMarkdown {{
                color: #F8FAFC !important;  /* Pristine Soft White */
                font-weight: 600 !important;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
            }}
            
            /* Keeps main titles outside the card bright and vivid */
            .main-title {{
                color: #FFFFFF !important;
                text-shadow: 2px 2px 10px rgba(0,0,0,0.9);
                font-weight: bold;
                text-align: center;

            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass

    st.markdown('<h1 class="main-title">COPD PREDICTOR</h1>', unsafe_allow_html=True) # Set the main title with a specific style
    #st.title(':blue[COPD PREDICTION]')

    st.write("### Enter Patient Details:")

    # Input Fields in two columns
    col1, col2 = st.columns(2)

    with col1:
        Age_input = st.text_input("Enter your age", "")
        st.write("Entered age is", Age_input)

        Gender_input = st.selectbox(
             "Enter your gender?",
             le1.classes_,
             index=None,
             placeholder="Select Gender..."
             )
        st.write("You selected:", Gender_input)


        BP_input = st.selectbox(
             "Enter your Blood pressure level?",
            le2.classes_,
            index=None,
            placeholder="Select Blood Pressure Level..."
            )
        st.write("You selected:", BP_input)

        Cholesterol_input = st.selectbox(
            "Enter your cholesterol level?",
            le3.classes_,
            index=None,
            placeholder="Select Cholesterol Level..."
            )
        st.write("You selected:", Cholesterol_input)

        Smoking_input = st.selectbox(
            "Do you smoke?",
            le4.classes_,
            index=None,
            placeholder="Select Smoking Status..."
            )
        st.write("You selected:", Smoking_input)

    with col2:
        Alcohol_input = st.selectbox(
            "Do you Consume Alcohol?",
            le5.classes_,
            index=None,
            placeholder="Select Yes/No..."
            )
        st.write("You selected:", Alcohol_input)
 
        Exercise_input = st.selectbox(
            "Do you Exercise regularly?",
            le6.classes_,
            index=None,
            placeholder="Select Yes/No..."
            )
        st.write("You selected:", Exercise_input)

        BMI_input = st.text_input("Enter your BMI", "")
        st.write("Entered BMI is", BMI_input)

        Familyhistory_input = st.selectbox(
            "Do anyone in your family has COPD?",
            le7.classes_,
            index=None,
            placeholder="Select Yes/No..."
            )
        st.write("You selected:", Familyhistory_input)


    pred= st.button('PREDICT')

    # Prediction Logic
    if pred:

        if Age_input and BMI_input:
            
            # Transform Categorical Data using loaded Encoders
            gender_enc = le1.transform([Gender_input])[0]
            bloodpressure_enc = le2.transform([BP_input])[0]
            cholesterol_enc = le3.transform([Cholesterol_input])[0]
            smoking_enc = le4.transform([Smoking_input])[0]
            alcoholconsumption_enc = le5.transform([Alcohol_input])[0]
            exercise_enc = le6.transform([Exercise_input])[0]
            familyhistory_enc = le7.transform([Familyhistory_input])[0]

            Features = [[
                float(Age_input),
                gender_enc,
                bloodpressure_enc,
                cholesterol_enc,
                smoking_enc,
                alcoholconsumption_enc,
                exercise_enc,
                float(BMI_input),
                familyhistory_enc
            ]]

            scaled = scalar.transform(Features)
            Prediction = model.predict(scaled)

            # Simulate processing time with a spinner
            with st.spinner("Wait for it...", show_time=True):
                time.sleep(2)
            st.success("Done!")

            # Convert the binary prediction (0 or 1) to a readable string
            result_text = "YES" if Prediction[0] == 1 else "NO"

            if Prediction[0]==1:
                st.error('COPD detected')

                st.snow()
            else:
                st.success('Congratulations! you dont have COPD')

                st.balloons()


            # Structural text building for data download
            report_content = f"""=====================================
    PATIENT REPORT       
=====================================
Age: {Age_input}
Gender: {Gender_input}
Blood Pressure: {BP_input}
Cholesterol: {Cholesterol_input}
Smoking Status: {Smoking_input}
Alcohol Consumption: {Alcohol_input}
Exercise: {Exercise_input}
BMI: {BMI_input}
Family History: {Familyhistory_input}
-------------------------------------
PREDICTED COPD STATUS: {result_text}
=====================================
Generated via Machine Learning Model
"""
       

            # Clean download placement below valuation block
            st.download_button(
                label="📥 Download Valuation Summary Report",
                data=report_content,
                file_name=f"valuation_{Age_input}_{Gender_input}.txt",
                mime="text/plain",
                use_container_width=True
            )

            # Feedback buttons
            sentiment_mapping = ["one", "two", "three", "four", "five"]
            selected = st.feedback("stars")
            if selected is not None:
                st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")

        else:

            # Warning if clicked predict without completing the form
            st.warning("Please enter Age and BMI!") 

if __name__ == '__main__':
    main()

