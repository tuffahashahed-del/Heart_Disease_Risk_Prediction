import pandas as pd
import streamlit as st
import sqlite3

# --- إنشاء أو الاتصال بقاعدة البيانات
conn = sqlite3.connect("heart_disease_data.db")
cursor = conn.cursor()

# --- إنشاء جدول إذا مش موجود
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_inputs (
        Age INTEGER,
        Sex INTEGER,
        ChestPainType INTEGER,
        RestingBP INTEGER,
        Cholesterol INTEGER,
        FastingBS INTEGER,
        RestingECG INTEGER,
        MaxHR INTEGER,
        ExerciseAngina INTEGER,
        Oldpeak REAL,
        ST_Slope INTEGER,
        Prediction INTEGER
    )
''')
conn.commit()

# --- واجهة المستخدم
st.title("Heart Disease Risk Prediction App 💡")
st.subheader("Enter your health details:")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", ["Female", "Male"])
pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'ASY'])
resting = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)
fasting = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
ECG = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
Max = st.number_input("Max Heart Rate", 60, 202, 150)
Exercise = st.selectbox("Exercise-Induced Angina", ['N', 'Y'])
peak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
Slope = st.selectbox("ST Slope", ["Up", 'Flat', 'Down'])

if st.button("Predict Risk 🎯"):
    # --- تجهيز الداتا
    input_data = {
        "Age": age,
        "Sex": 1 if sex == "Male" else 0,
        "ChestPainType": {"ATA": 1, "NAP": 2, "ASY": 3}[pain],
        "RestingBP": resting,
        "Cholesterol": cholesterol,
        "FastingBS": fasting,
        "RestingECG": {'Normal': 0, 'ST': 1, 'LVH': 2}[ECG],
        "MaxHR": Max,
        "ExerciseAngina": 1 if Exercise == "Y" else 0,
        "Oldpeak": peak,
        "ST_Slope": {"Up": 0, 'Flat': 1, 'Down': 2}[Slope]
    }

    input_df = pd.DataFrame([input_data])

    # --- عرض النتيجة
    st.success(f"Prediction Result: {'High Risk ❤‍🔥' if prediction == 1 else 'Low Risk 💚'}")

    # --- تخزين بالقاعدة
    cursor.execute('''
        INSERT INTO user_inputs (
            Age, Sex, ChestPainType, RestingBP, Cholesterol,
            FastingBS, RestingECG, MaxHR, ExerciseAngina,
            Oldpeak, ST_Slope, Prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        input_data["Age"], input_data["Sex"], input_data["ChestPainType"],
        input_data["RestingBP"], input_data["Cholesterol"], input_data["FastingBS"],
        input_data["RestingECG"], input_data["MaxHR"], input_data["ExerciseAngina"],
        input_data["Oldpeak"], input_data["ST_Slope"], prediction
    ))
    print("Saving to DB:", input_data, "Prediction:", prediction)
    conn.commit()
    st.info("✔ Data has been saved successfully to the database.")