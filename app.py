import streamlit as st
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide",
    )
st.title("👨‍💼Employee Attrition Predictor")
st.write("ENTER EMPLOYEE DETAILS BELOW TO ESTIMATE THE LIKELIHOOD OF EMPLOYEE ATTRITION")

Age=st.number_input("Age",min_value=18,max_value=100,value=30)
Daily_rate=st.number_input("Daily rate",min_value=0,value=800)
Distance_from_home=st.number_input("Distance from home",min_value=1,max_value=29,value=10)
Education=st.number_input("Education",min_value=1,max_value=5,value=3)
Employee_count=1
Environmental_satisfaction=st.number_input("Environmental Satisfaction",min_value=1,max_value=4,value=3)
Hourly_rate=st.number_input("Hourly rate",min_value=30,max_value=100,value=60)
Job_Involvement=st.number_input("Job involvement",min_value=1,max_value=4,value=3)
Job_level=st.number_input("Job level",min_value=1,max_value=5,value=2)
Job_satisfaction=st.number_input("Job satisfaction", min_value=1, max_value=4, value=3)
Monthly_income=st.number_input("Monthly income", min_value=1009, max_value=19999, value=5000)
Monthly_rate=st.number_input("Monthly rate", min_value=2094, max_value=26999, value=14000)
Num_companies_worked=st.number_input("Number of companies worked", min_value=0, max_value=9, value=2)
Percent_salary_hike=st.number_input("Percent salary hike", min_value=11, max_value=25, value=15)
Performance_rating=st.number_input("Performance rating", min_value=3, max_value=4, value=3)
Relationship_satisfaction=st.number_input("Relationship satisfaction", min_value=1, max_value=4, value=3)
Stock_option_level=st.number_input("Stock option level", min_value=0, max_value=3, value=1)
Total_working_years=st.number_input("Total working years", min_value=0, max_value=40, value=8)
Training_times_last_year=st.number_input("Training times last year", min_value=0, max_value=6, value=3)
Work_life_balance=st.number_input("Work life balance", min_value=1, max_value=4, value=3)
Years_at_company=st.number_input("Years at company", min_value=0, max_value=40, value=5)
Years_in_current_role=st.number_input("Years in current role", min_value=0, max_value=18, value=3)
Years_since_last_promotion=st.number_input("Years since last promotion", min_value=0, max_value=15, value=1)
Years_with_current_manager=st.number_input("Years with current manager", min_value=0, max_value=17, value=3)
business_travel=st.selectbox("business Travel",
                             ["Travel_Rarely","Travel_Frequently","Non-Travel"]
                             )
department=st.selectbox("Department",
                        ["Sales","Research & Development","human Resources"])
education_field = st.selectbox(
    "Education Field",
    ["Life Sciences", "Medical", "Marketing",
     "Technical Degree", "Human Resources", "Other"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

job_role = st.selectbox(
    "Job Role",
    ["Sales Executive", "Research Scientist",
     "Laboratory Technician", "Manufacturing Director",
     "Healthcare Representative", "Manager",
     "Sales Representative", "Research Director",
     "Human Resources"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)
import joblib
import pandas as pd
model=joblib.load("logistic_model.pkl")
scaler=joblib.load("scaler.pkl")
input_data = pd.DataFrame({
    "Age": [Age],
    "DailyRate": [Daily_rate],
    "DistanceFromHome": [Distance_from_home],
    "Education": [Education],
    "EmployeeCount": [1],
    "EmployeeNumber": [0],
    "EnvironmentSatisfaction": [Environmental_satisfaction],
    "HourlyRate": [Hourly_rate],
    "JobInvolvement": [Job_Involvement],
    "JobLevel": [Job_level],
    "JobSatisfaction": [Job_satisfaction],
    "MonthlyIncome": [Monthly_income],
    "MonthlyRate": [Monthly_rate],
    "NumCompaniesWorked": [Num_companies_worked],
    "PercentSalaryHike": [Percent_salary_hike],
    "PerformanceRating": [Performance_rating],
    "RelationshipSatisfaction": [Relationship_satisfaction],
    "StandardHours": [80],
    "StockOptionLevel": [Stock_option_level],
    "TotalWorkingYears": [Total_working_years],
    "TrainingTimesLastYear": [Training_times_last_year],
    "WorkLifeBalance": [Work_life_balance],
    "YearsAtCompany": [Years_at_company],
    "YearsInCurrentRole": [Years_in_current_role],
    "YearsSinceLastPromotion": [Years_since_last_promotion],
    "YearsWithCurrManager": [Years_with_current_manager]
})
categorical_data = pd.DataFrame({
    "BusinessTravel": [business_travel],
    "Department": [department],
    "EducationField": [education_field],
    "Gender": [gender],
    "JobRole": [job_role],
    "MaritalStatus": [marital_status],
    "OverTime": [overtime]
})

categorical_data=pd.get_dummies(categorical_data,drop_first=True)
input_data=pd.concat([input_data,categorical_data],axis=1)
import joblib
import pandas as pd
model=joblib.load("logistic_model.pkl")
scaler=joblib.load("scaler.pkl")
feature_columns=joblib.load("feature_columns.pkl")
input_data=input_data.reindex(
    columns=feature_columns,
    fill_value=0
    )
input_scaled=scaler.transform(input_data)
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Employee is likely to leave — Risk: {probability:.1%}")
    else:
        st.success(f"Employee is likely to stay — Risk of leaving: {probability:.1%}")
 

    
    
     
