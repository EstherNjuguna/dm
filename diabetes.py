import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
st.markdown(
    """
    <style>
    .title {
        color: #4CAF50;
        font-size: 32px;
        text-align: center;
        font-weight: bold;
    }
    .input {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 8px;
        color: #4CAF50;
    }
    .stTextInput>div>input {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        color: green;
    }
    .stNumberInput>div>input {
        border: 2px solid #4CAF50;
        border-radius: 5px;
        color: green;
    }
        /* General Text Styling */
    body, .stMarkdown, .stTextInput, .stNumberInput, .stButton {
        color: #4CAF50; /* Green text for most elements */


    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Diabetes and Depression Risk scores</div>', unsafe_allow_html=True)


# Load the trained model and scaler
try:
    model = joblib.load('random_forest_resampled.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure the file is uploaded.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading the model: {e}")
    st.stop()

try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Scaler file not found. Please ensure the file is uploaded.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading the scaler: {e}")
    st.stop()



# Diabetes Prediction Section
st.header("Diabetes Prediction")

# Define the selected features based on the model's training
selected_features = ['age', 'weight_kg', 'height_cm', 'bmi', 'sys_bp', 'dia_bp', 'glucose']

# Input form for user data (Diabetes)
with st.form("user_input_form"):
    age = st.number_input("Age", min_value=0, step=1)
    weight_kg = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
    height_cm = st.number_input("Height (cm)", min_value=0.0, step=0.1)
    
    # Calculate BMI dynamically
    if height_cm > 0:
        height_m = height_cm / 100  # Convert height to meters
        bmi = weight_kg / (height_m ** 2)
        st.write(f"Calculated BMI: {bmi:.2f}")
    else:
        bmi = 0
        st.error("Height must be greater than 0 to calculate BMI.")

    sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=0.0, step=0.1)
    dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0.0, step=0.1)
    glucose = st.number_input("Glucose Level ((µmol/L)", min_value=0.0, step=0.1)

    # Submit button for diabetes prediction
    submitted = st.form_submit_button("Predict Diabetes Risk")

if submitted:
    # Validate user inputs
    if weight_kg <= 0 or height_cm <= 0 or bmi <= 0:
        st.error("Weight, height, and BMI must be positive values.")
    elif sys_bp < 0 or dia_bp < 0:
        st.error("Blood pressure values cannot be negative.")
    else:
        # Prepare the input data for diabetes prediction, using only the selected features
        input_data = pd.DataFrame({
            'age': [age],
            'weight_kg': [weight_kg],
            'height_cm': [height_cm],
            'bmi': [bmi],
            'sys_bp': [sys_bp],
            'dia_bp': [dia_bp],
            'glucose': [glucose]
        })
        
        # Apply the same scaling as was done during training
        input_data_scaled = scaler.transform(input_data[selected_features])  # Only scale the selected features
        
        # Make prediction for diabetes
        prediction = model.predict(input_data_scaled)
        
        # Display the result for diabetes prediction
        if prediction[0] == 1:
            st.success("The model predicts a risk of diabetes.")
        else:
            st.success("The model predicts no risk of diabetes.")

# Depression Risk Assessment Section (PH9)
st.header("Depression Risk Assessment (PH9)")

# Section for PH9 questions
st.subheader("Please answer the following questions:")

# PH9 Questions
ph9_answers = {
    "Little interest or pleasure in doing things?": st.radio("1. Little interest or pleasure in doing things?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Feeling down, depressed, or hopeless?": st.radio("2. Feeling down, depressed, or hopeless?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Trouble falling or staying asleep, or sleeping too much?": st.radio("3. Trouble falling or staying asleep, or sleeping too much?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Feeling tired or having little energy?": st.radio("4. Feeling tired or having little energy?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Poor appetite or overeating?": st.radio("5. Poor appetite or overeating?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Feeling bad about yourself, or that you are a failure, or have let yourself or your family down?": st.radio("6. Feeling bad about yourself, or that you are a failure, or have let yourself or your family down?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Trouble concentrating on things, such as reading the newspaper or watching television?": st.radio("7. Trouble concentrating on things, such as reading the newspaper or watching television?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?": st.radio("8. Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual?", ("Not at all", "Several days", "More than half the days", "Nearly every day")),
    "Thoughts that you would be better off dead, or of hurting yourself in some way?": st.radio("9. Thoughts that you would be better off dead, or of hurting yourself in some way?", ("Not at all", "Several days", "More than half the days", "Nearly every day"))
}

# Mapping responses to numeric values for scoring
score_map = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

# Calculate the total score based on responses
if st.button("Assess Depression Risk"):
    ph9_score = sum([score_map[answer] for answer in ph9_answers.values()])
    
    # Depression Risk Classification based on PH9 score
    if ph9_score < 5:
        risk_level = "Low risk of depression."
    elif 5 <= ph9_score < 15:
        risk_level = "Moderate risk of depression."
    else:
        risk_level = "High risk of depression."
    
    # Display the result for depression risk assessment
    st.write(f"Total PH9 Score: {ph9_score}")
    st.write(risk_level)
