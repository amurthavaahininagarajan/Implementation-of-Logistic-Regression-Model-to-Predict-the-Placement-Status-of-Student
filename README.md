# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Find the null and duplicate values.
3. Using logistic regression find the predicted values of accuracy , confusion matrices
4. Display the results

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AMURTHA VAAHINI.KN
RegisterNumber:  212222240008
**
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

*/
```

## Output:
## 1.PLACEMENT DATA
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/3aaf36a3-3225-4e02-894b-679fae92571c)

## 2.SALARY DATA:
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/bfb3690b-0316-4a6a-8d41-eec8657f928c)

## 3.Checking the null function()
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/821980ec-0828-4d23-a523-210744cfe740)

## 4.Data Duplicate
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/7dc60349-0dc5-465b-8865-346198a9851d)

## 5.Print Data
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/973ca213-b6a0-47d6-a04c-42f6152a4fb3)
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/b82f1f41-495c-4284-9d63-7dd9a8f1f74e)

## 6.Data Status
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/a308db6f-3645-4a26-bad3-854a0dd8df13)

## 7.y_prediction array
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/68313695-f848-4c88-9b28-cbc943b46b2c)

## 8.Accuracy value
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/4fd084e5-a1a4-4f2b-90f9-2272afecb5a1)

## 9.Confusion matrix
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/e89e4d54-4861-49a2-ae42-3855314d2c65)

## 10.Classification Report
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/11220784-7d11-4818-86f1-529911748b78)

## 11.Prediction of LR
![image](https://github.com/amurthavaahininagarajan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118679102/8c568904-1ce8-492c-8294-4cc087c0cdcc)






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
