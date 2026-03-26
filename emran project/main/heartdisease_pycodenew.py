import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')
data = pd.read_csv('C:\\Users\\USER\\PycharmProjects\\PythonProject\\emran project\\main\\heartc.csv')
print(data)
print(data.shape)
print(data.columns)
print(data.info())
sns.heatmap(data.isnull(),cmap = 'magma',cbar = False);#No null values in the data
print(data.describe().T)
print('********************************')
yes = data[data['HeartDisease'] == 1].describe().T
no = data[data['HeartDisease'] == 0].describe().T
colors = ['#F93822','#FDD20E']
fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (5,5))
plt.subplot(1,2,1)
sns.heatmap(yes[['mean']],annot = True,cmap = colors,
            linewidths = 0.4,linecolor = 'black',cbar = False,fmt = '.2f',)
plt.title('Heart Disease');
plt.subplot(1,2,2)
sns.heatmap(no[['mean']],annot = True,cmap = colors,
            linewidths = 0.4,linecolor = 'black',cbar = False,fmt = '.2f')
plt.title('No Heart Disease');
fig.tight_layout(pad = 2)#Mean values of all the features for cases of heart diseases and non-heart diseases.
#Dividing features into Numerical and Categorical
col = list(data.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(data[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)
print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)
#Here, categorical features are defined if the the attribute has less than 6 unique elements else it is a numerical feature.
#Typical approach for this division of features can also be based on the datatypes of the elements of the respective attribute.
#Eg : datatype = integer, attribute = numerical feature ; datatype = string, attribute = categorical feature
#For this dataset, as the number of features are less, we can manually check the dataset as well.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = data.copy(deep = True)
#Distribution of Categorical Features 
df1['Sex'] = le.fit_transform(df1['Sex'])
df1['ChestPainType'] = le.fit_transform(df1['ChestPainType'])
df1['RestingECG'] = le.fit_transform(df1['RestingECG'])
df1['ExerciseAngina'] = le.fit_transform(df1['ExerciseAngina'])
df1['ST_Slope'] = le.fit_transform(df1['ST_Slope'])
fig, ax = plt.subplots(nrows = 3,ncols = 2,figsize = (10,15))
for i in range(len(categorical_features) - 1):
    
    plt.subplot(3,2,i+1)
    sns.distplot(df1[categorical_features[i]],kde_kws = {'bw' : 1},color = colors[0]);
    title = 'Distribution : ' + categorical_features[i]
    plt.title(title)
    
plt.figure(figsize = (4.75,4.55))
sns.distplot(df1[categorical_features[len(categorical_features) - 1]],kde_kws = {'bw' : 1},color = colors[0])
title = 'Distribution : ' + categorical_features[len(categorical_features) - 1]
plt.title(title);#All the categorical features are near about Normally Distributed.
print("***************************************")
#Distribution of Numerical Features
fig, ax = plt.subplots(nrows = 2,ncols = 2,figsize = (10,9.75))
for i in range(len(numerical_features) - 1):
    plt.subplot(2,2,i+1)
    sns.distplot(data[numerical_features[i]],color = colors[0])
    title = 'Distribution : ' + numerical_features[i]
    plt.title(title)
plt.show()

plt.figure(figsize = (4.75,4.55))
sns.distplot(df1[numerical_features[len(numerical_features) - 1]],kde_kws = {'bw' : 1},color = colors[0])
title = 'Distribution : ' + numerical_features[len(numerical_features) - 1]
plt.title(title);#Oldpeak's data distribution is rightly skewed.
#Cholestrol has a bidmodal data distribution.

#data scaling(Machine learning model does not understand the units of the values of the features. 
#It treats the input just as a simple number but does not understand the true meaning of that value.it becomes necessary to scale the data.
#Eg : Age = Years; FastingBS = mg / dl)
from sklearn.preprocessing import MinMaxScaler,StandardScaler
mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization

df1['Oldpeak'] = mms.fit_transform(df1[['Oldpeak']])
df1['Age'] = ss.fit_transform(df1[['Age']])
df1['RestingBP'] = ss.fit_transform(df1[['RestingBP']])
df1['Cholesterol'] = ss.fit_transform(df1[['Cholesterol']])
df1['MaxHR'] = ss.fit_transform(df1[['MaxHR']])
print(df1.head())

#Modeling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, RocCurveDisplay, classification_report, accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. تحميل البيانات (تأكد من وجود df1)
# df1 = pd.read_csv('your_data.csv')

# 2. تحضير البيانات
features = df1.drop(['HeartDisease', 'RestingBP', 'RestingECG'], axis=1)
target = df1['HeartDisease']

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=2)

# 4. تحجيم البيانات (مهم لخوارزمية KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. دالة التقييم المعدلة
def model(classifier, X_train, X_test, y_train, y_test):
    # تدريب النموذج
    classifier.fit(X_train, y_train)
    
    # التنبؤات
    y_pred = classifier.predict(X_test)
    
    # حساب المقاييس
    accuracy = accuracy_score(y_test, y_pred)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=cv, scoring='roc_auc')
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # عرض النتائج
    print("="*50)
    print(f"{'KNN Classifier Results':^50}")
    print("="*50)
    print(f"{'Accuracy:':<25} {accuracy:.2%}")
    print(f"{'Cross Validation (AUC):':<25} {cv_scores.mean():.2%}")
    print(f"{'ROC AUC Score:':<25} {roc_auc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # رسم منحنى ROC
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(classifier, X_test, y_test)
    plt.title('ROC Curve - KNN Classifier')
    plt.show()
    
    # رسم مصفوفة الارتباك
    cm = confusion_matrix(y_test, y_pred)
    names = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    counts = cm.flatten()
    percentages = [f"{x/cm.sum():.2%}" for x in counts]
    labels = [f"{n}\n{c}\n{p}" for n, c, p in zip(names, counts, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    plt.title('Confusion Matrix - KNN Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return classifier

# 6. إنشاء وتقييم النموذج
classifier_knn = KNeighborsClassifier(
    leaf_size=1,
    n_neighbors=3,
    p=1,  # المسافة من نوع Manhattan (p=1)
    metric='minkowski'
)

# 7. تشغيل النموذج مع البيانات المحجمة
trained_knn = model(classifier_knn, X_train_scaled, X_test_scaled, y_train, y_test)

# حفظ النموذج المدرب
joblib.dump(trained_knn, r"C:\Users\USER\PycharmProjects\PythonProject\emran project\main\model.pkl")

# حفظ الـ Scaler المستخدم
joblib.dump(scaler, r"C:\Users\USER\PycharmProjects\PythonProject\emran project\main\scaler.pkl")


# 7. تشغيل النموذج مع البيانات المحجمة
"""
trained_knn = model(classifier_knn, X_train_scaled, X_test_scaled, y_train, y_test)
This dataset is great for understanding how to handle binary classification problems with the combination of numerical and categorical features.
Subject matter experts, in this case doctors or nurses, can be assisted by providing insights that enables them to take the next line of action.
For feature engineering, it might feel confusing about the order of the processes. In this case, data scaling was executed before the feature selection test. We might feel like we are tampering the data before passing it to the tests but the results are same irrespective of the order of the process. (Try it out!)
For this problem, outlier detection was not done as I was not able to read any papers about heart diseases. It becomes a pivotal part to understand the subject before removing outliers even though the outlier detection tests come out positive.
Visualization is key. It makes the data talkative. Displaying the present information and results of any tests or output through visualization becomes crucial as it makes the understanding easy.
For modeling, hyperparameter tuning is not done. It can push the performances of the algorithms. Overall the algorithm performances are good.
"""

import joblib

model = joblib.load(r"C:\Users\USER\PycharmProjects\PythonProject\emran project\main\model.pkl")
scaler = joblib.load(r"C:\Users\USER\PycharmProjects\PythonProject\emran project\main\scaler.pkl")












