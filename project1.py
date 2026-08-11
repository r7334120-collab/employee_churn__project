import pandas as pd
df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(df.head())
#cleaning rows
print(df.info())
print(df.isnull().sum())
#converting categorical into numbers
print(df.dtypes)
df["Attrition"]=df["Attrition"].map({"Yes":1,"No":0})
df=pd.get_dummies(df,drop_first=True)
print(df.head())
#Separating X and y
X=df.drop("Attrition",axis=1)
import joblib
joblib.dump(X.columns.tolist(),"feature_columns.pkl")
y=df["Attrition"]
print(X.shape)
print(y.shape)
#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#scaling data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
#training data
from sklearn.linear_model import LogisticRegression
logistic_model=LogisticRegression(max_iter=100)
logistic_model.fit(X_train_scaled,y_train)
logistic_prediction=logistic_model.predict(X_test_scaled)
print("Prediction counts:")
print(pd.Series(logistic_prediction).value_counts())
print(logistic_prediction)
#evualation
from sklearn.metrics import accuracy_score
accuracy1=accuracy_score(y_test,logistic_prediction)
print("Accuracy=",accuracy1)
from sklearn.metrics import precision_score,recall_score
Precision1=precision_score(y_test,logistic_prediction,pos_label=1)
Recall1=recall_score(y_test,logistic_prediction,pos_label=1)
fl_score1=2*(Precision1*Recall1)/(Precision1+Recall1)
print("Precision=",Precision1)
print("Recall=",Recall1)
print("Fl_score=",fl_score1)
#trying other model
from sklearn.tree import DecisionTreeClassifier
decision_model=DecisionTreeClassifier(max_depth=10)
decision_model.fit(X_train_scaled,y_train)
decision_prediction=decision_model.predict(X_test_scaled)
print(decision_prediction)
#evaluation
from sklearn.metrics import accuracy_score
accuracy2=accuracy_score(y_test,decision_prediction)
print("Accuracy=",accuracy2)
from sklearn.metrics import precision_score,recall_score
Precision2=precision_score(y_test,decision_prediction,pos_label=1)
Recall2=recall_score(y_test,decision_prediction,pos_label=1)
fl_score2=2*(Precision2*Recall2)/(Precision2+Recall2)
print("Precision=",Precision2)
print("Recall=",Recall2)
print("fl_score=",fl_score2)
#Logisitic Model finalized
print(X.columns.tolist())
#predicting leave for new employee
new_employee=X.iloc[[0]]
print(new_employee)
#scaling new employee
new_employee_scaled=scaler.transform(new_employee)
new_employee_prediction=logistic_model.predict(new_employee_scaled)
print("Prediction=",new_employee_prediction)
print("Actual=",y.iloc[0])
print("Predicted=",new_employee_prediction[0])
import joblib
joblib.dump(logistic_model,"logistic_model.pkl")
joblib.dump(scaler,"scaler.pkl")






