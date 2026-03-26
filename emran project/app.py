#SHAHD TUFFAHA
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
# تحديد المسار الرئيسي للملفات
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "main", "model.pkl")
scaler_path = os.path.join(base_path, "main", "scaler.pkl")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
image_path = os.path.join(base_path, "image", "aiheart.jpg")
st.image(image_path, use_container_width=True)
st.markdown("<h3 style='text-align: center; color: #2980B9;'>Powered by AI for Your Health 💙</h3>", unsafe_allow_html=True)
st.markdown("""<style>.stApp {background-color:Azure;}</style>""",
    unsafe_allow_html=True)
st.write('''
<h2 style="color: #2980B9;font-size:16px;">
Welcome to the Heart Disease Risk Prediction App💡<br>
Enter your health details to predict your heart disease risk using AI🤖<br>
Your well-being is our priority ....❤️<br>
</h2>
''', unsafe_allow_html=True)

c1,c2,c3=st.columns([2,2,2])
with c1:
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">Age🔎</h1>', unsafe_allow_html=True)
    age=st.number_input('Age',18,100,label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">Sex</h1>', unsafe_allow_html=True)
    Sex=st.selectbox('Sex',['Female',"Male"],label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">Chest Pain Type🩻</h1>', unsafe_allow_html=True)
    pain=st.selectbox('ChestPainType',['TA','ATA','NAP','ASY'],label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">RestingBP🩺</h1>', unsafe_allow_html=True)
    resting=st.number_input("RestingBP",80,200,120,label_visibility="hidden")
with c2:
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">Cholesterol🧪 </h1>', unsafe_allow_html=True)
    Cholesterol=st.number_input('Cholesterol',100,400,200,label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">Fasting Blood Sugar🩸</h1>', unsafe_allow_html=True)
    fasting=st.selectbox("FastingBS",[0,1],label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">RestingECG📈</h1>', unsafe_allow_html=True)
    ECG=st.selectbox('RestingECG',['Normal','ST','LVH'],label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">MaxHeart Rate💓</h1>', unsafe_allow_html=True)
    Max=st.number_input("MaxHR",60,202,150,label_visibility="hidden")
with c3:
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">ExerciseAngina🏃🏻</h1>', unsafe_allow_html=True)
    Exercise=st.selectbox('ExerciseAngina',['N','Y'],label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">Oldpeak📉</h1>', unsafe_allow_html=True)
    peak=st.number_input("Oldpeak",0.0,10.0,1.0,label_visibility="hidden")
    st.markdown('<h1 style="color:SteelBlue;font-size:16px; text-align:left;">ST_Slope📊</h1>', unsafe_allow_html=True)
    Slope=st.selectbox("ST_Slope",["Up",'Flat','Down'],label_visibility="hidden")
    with st.expander("ℹ️ Help:"):
        st.markdown("""
           *Age: Patient's age in years.

           *Sex: Biological sex of the patient."0"= Female,"1"= Male.

           *Chest Pain Type (cp):
           - 'TA': typical angina
           
           - 'ATA': Atypical angina
           - 'NAP': Non-anginal pain
           - 'ASY': Asymptomatic

           *RestingBP: Resting blood pressure (in mm Hg) when patient is seated.

           *Cholesterol: Serum cholesterol in mg/dl.

           *FastingBS: Fasting blood sugar > 120 mg/dl. '1' = true, '0' = false.

           *RestingECG: Results of resting electrocardiogram:
           - 'Normal': Normal
           - 'ST': ST-T wave abnormality
           - 'LVH': Left ventricular hypertrophy

           *MaxHR: Maximum heart rate achieved during exercise.

           *ExerciseAngina: Exercise-induced angina. 'Y' = Yes, 'N' = No.

           *Oldpeak: ST depression induced by exercise relative to rest.

           **T_Slope: Slope of the peak exercise ST segment:
           - "Up": Upsloping
           - 'Flat': Flat
           - 'Down': Downsloping
           """)

with c3:
    st.markdown('<br>',unsafe_allow_html=True)
    if st.button('Predict Risk🎯'):
        input_data = {
            "Age": age,
            "Sex": 1 if Sex == "Male" else 0,
            "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2,'ASY':3}[pain],
            "RestingBP": resting,
            "Cholesterol": Cholesterol,
            "FastingBS": fasting,
            "RestingECG": {'Normal': 0, 'ST': 1, 'LVH': 2}[ECG],
            "MaxHR": Max,
            "ExerciseAngina": 1 if Exercise == "Y" else 0,
            "Oldpeak": peak,
            "ST_Slope": {"Up": 0, 'Flat': 1, 'Down': 2}[Slope]

        }
        input_df = pd.DataFrame([input_data])
        input_df = input_df[scaler.feature_names_in_]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        risk_percent = round(probability * 100, 2)
# بدايه الداتا بيز
        import sqlite3

        # فتح اتصال بقاعدة البيانات
        conn = sqlite3.connect("heart_disease_data.db")
        cursor = conn.cursor()
        prediction_text = "High Risk" if prediction == 1 else "Low Risk"
        # إنشاء جدول إذا ما كان موجود
        cursor.execute('''
                            CREATE TABLE IF NOT EXISTS user_inputs (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                                Prediction TEXT
                            )
                        ''')

        # إدخال البيانات في الجدول
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
            input_data["Oldpeak"], input_data["ST_Slope"], prediction_text
        ))
        conn.commit()
        conn.close()
        # نهايه الداتا بيز
        #القلب
        labels = ['Risk', 'Safe']
        sizes = [risk_percent, 100 - risk_percent]
        colors = ['#E74C3C', '#2ECC71']  # أحمر للخطر، أخضر للأمان

        t = np.linspace(0, 2 * np.pi, 1000)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.fill(x, y, color='lightgray')  # خلفية القلب بلون رمادي فاتح

        y_limit = y.min() + (y.max() - y.min()) * (risk_percent / 100)
        mask = y <= y_limit

        # تلوين الجزء السفلي حسب نسبة الخطر
        ax.fill_between(x, y_limit, y, where=mask, color='#E74C3C')

        ax.set_aspect('equal')
        ax.axis('off')  # إخفاء المحاور

        # إضافة النص داخل القلب
        ax.text(0, 0, f"{risk_percent}%", fontsize=40, color='white', fontweight='bold', ha='center', va='center')
        st.pyplot(fig)

        # عرض النتيجة نصياً أيضاً
        st.markdown(f"<h3 style='color:#1F618D;'>Risk Probability:: {risk_percent}%</h3>", unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"<h3 style='color:#C0392B;'>⚠️ High Risk</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color:#27AE60;'>✅ Low Risk</h3>", unsafe_allow_html=True)
        st.markdown("If you're at high risk, please consider consulting your doctor. This tool is for educational purposes only.",unsafe_allow_html=True)
#القلب
#الزو وتعديلاته
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: LightCoral; 
        color: white;               
        height:100px;               
        width: 200px;               
        border-radius:12px;        
        font-size: 50px;            
        font-weight: bold;          
        box-shadow: 3px 3px 5px grey; 
        transition: background-color 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #a62834; 
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)



