import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# تحميل البيانات
df = pd.read_csv("C:\\Users\\USER\\PycharmProjects\\PythonProject\\emran project\\main\\heartc.csv")  # تأكدي من اسم الملف ومكانه

# تحويل البيانات النصية إلى أرقام (مثل ما كنتِ عاملة بالمشروع الأصلي)
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2})
df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

# تحديد الميزات والهدف
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# تعويض القيم المفقودة (Imputation)
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# تقييس البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# تدريب نموذج KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# حفظ الموديل والمقيّس والمعوّض
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ تم تدريب النموذج بنجاح وحفظ الملفات.")
