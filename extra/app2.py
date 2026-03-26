import pandas as pd
import streamlit as st
from click import style
from numpy.f2py.symbolic import as_ge
st.image("images/darkk.jpg",use_container_width=True)
st.markdown("""<style>.stApp {background-color:#17202a
;}</style>""",
    unsafe_allow_html=True)


st.write('''
<h2 style="font-size:16px;">
Welcome to the Heart Disease Risk Prediction App💡<br>
Enter your health details to predict your heart disease risk using AI🤖<br>
Your well-being is our priority ....❤️<br>
</h2>
''', unsafe_allow_html=True)

c1,c2,c3=st.columns([2,2,2])
with c1:
    st.markdown('<h1 style="font-size:16px; text-align:left;">Age🔎</h1>', unsafe_allow_html=True)
    age=st.number_input('',18,100)
    st.markdown('<h1 style="font-size:16px; text-align:left;">sex</h1>', unsafe_allow_html=True)
    sex=st.selectbox('',['Female',"Male"])
    st.markdown('<h1 style="font-size:16px; text-align:left;">Chest Pain Type🩻</h1>', unsafe_allow_html=True)
    pain=st.selectbox('',['ATA','NAP','ASY'])
    st.markdown('<h1 style="font-size:16px; text-align:left;">Resting Blood Pressure🩺</h1>', unsafe_allow_html=True)
    resting=st.number_input("",80,200,120)
with c2:
    st.markdown('<h1 style="font-size:16px; text-align:left;">Cholestrol🧪 </h1>', unsafe_allow_html=True)
    Cholestrol=st.number_input('',100,400,200)
    st.markdown('<h1 style="font-size:16px; text-align:left;">Fasting Blood Sugar🩸</h1>', unsafe_allow_html=True)
    fasting=st.selectbox("",[0,1])


    st.markdown('<h1 style="font-size:16px; text-align:left;">RestingECG📈</h1>', unsafe_allow_html=True)
    ECG=st.selectbox('',['Normal','ST','LVH'])
    st.markdown('<h1 style="font-size:16px; text-align:left;">MaxHeart Rate💓</h1>', unsafe_allow_html=True)
    Max=st.number_input("",60,202,150)
with c3:
    st.markdown('<h1 style="font-size:16px; text-align:left;">Exercise_Angina🏃🏻</h1>', unsafe_allow_html=True)
    Exercise=st.selectbox('',['N','Y'])
    st.markdown('<h1 style="font-size:16px; text-align:left;">Oldpeak📉</h1>', unsafe_allow_html=True)
    peak=st.number_input("",0.0,10.0,1.0)
    st.markdown('<h1 style="font-size:16px; text-align:left;">ST_Slope📊</h1>', unsafe_allow_html=True)
    Slope=st.selectbox("",["Up",'Flat','Down'])


with c3:
    st.markdown('<br>',unsafe_allow_html=True)
    if st.button('Predict Risk🎯'):
        input_data = {
            "Age": age,
            "Sex": 1 if sex == "Male" else 0,
            "ChestPainType": {"TA": 0, "ATA": 1, "NAP": 2, "ASY": 3}[pain],
            "RestingBP": resting,
            "Cholesterol": cholesterol,
            "Fasting Blood Sugar": fasting,
            "RestingECG": {'Normal': 0, 'ST': 1, 'LVH': 2}[ECG],
            "MaxHR": Max,
            "Exercise_Angina": 1 if Exercise == "Y" else 0,
            "Oldpeak": peak,
            "ST_Slope": {"Up": 0, 'Flat': 1, 'Down': 2}[Slope]

        }
        input_df = pd.DataFrame([input_data])
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #e63946;  
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
        background-color: #a62834;  /* لون مختلف عند التمرير */
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)