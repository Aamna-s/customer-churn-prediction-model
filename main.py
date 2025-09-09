import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from openai import OpenAI
import util as ut
client = OpenAI(
  base_url="https://api.groq.com/openai/v1",
  api_key = os.getenv("CHURN_CUSTOMER_API_KEY")
)

def explain_predictions(probability,input_dic, surname):

  prompt = f"""You are an expert data scientist at a bank, where you specialize in 
  interpreting and explaining predictions of machine learning models.
  
  Your machine learning model has predicted that a customer named {surname} has a 
  {round(probability * 100, 1)}% probability ofchurning, based on the information provided below:
  Here is the customer's information: {input_dic}
  I
  Here are the machine learning model's top 10 most important features for predicting churn:
                Feature Importance
          ----------------------------
          NumOfProducts     | 0.323888
          IsActiveMember    | 0.164146
          Age               | 0.109550
          Geography Germany | 0.091373
          Balance           | 0.052786
          Geography_France  | 0.046463
          Gender Female     | 0.045283
          Geography Spain   | 0.036855 
          CreditScore       | 0.035005 
          EstimatedSalary   | 0.032655 
          HasCrCard         | 0.031940
          Tenure            | 0.030054 
          Gender_Male       | 0.000000
          
  {pd.set_option('display.max_columns', None)}
  Here are summary statistics for churned customers: 
    {df[df['Exited'] == 1].describe()}

  Here are summary statistics for non-churned customers: 
    {df[df['Exited']==0].describe()}

  
  If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  -
  If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
  -
  Your explanation should be based on the customer's information, the summary statistics
  of churned and non-churned customers, and the feature importances provided.
  Don't mention the probability of churning, or the machine learning model, or say anything like 
  "Based on the machine learning model's prediction and top 10 most important features",
  just explain the prediction."""
  
  print("EXPLANATION PROMPT", prompt)
  raw_response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{
    "role": "user",
    "content": prompt
    }]
  )
  return raw_response.choices[0].message.content

def generate_email(probability,input_dict,surname,explanation):

  prompt = f"""You are a manager at HS Bank. You are responsible for ensuring customers
    stay with the bank and are incentivized with various offers.
    You noticed a customer named {surname} has a 
    {round(probability * 100, 1)}% probability of churning.
    
    Here is the customer's information:
    {input_dict}
    
    Here is some explanation as to why the customer might be at risk of churning:
    {explanation}
    
    Generate an email to the customer based on their information, asking them to stay 
    if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.
    Make sure to list out a set of incentives to stay based on their information, 
    in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer."""
  
  raw_response = client.chat.completions.create(
  model="llama-3.1-8b-instant",
  messages=[{
  "role": "user",
  "content": prompt
  }],)
  print("\n\nEMAIL PROMPT", prompt)
  return  raw_response.choices[0].message.content



def load_model(filename):
    with open(filename, 'rb') as file:
        return joblib.load(file)


def prepare_input(credit_score, age, tenure, balance, num_of_products,
                  has_cr_card, is_active_member, estimated_salary, gender,
                  location):
    input_dic = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Female": 1 if gender == "Female" else 0
    }
    input_df = pd.DataFrame([input_dic])
    input_df["CLV"] = input_df["Balance"] * input_df["NumOfProducts"]  
    input_df["TenureAgeRatio"] = input_df["Tenure"] / input_df["Age"]
  
    # Example age groups
    input_df["AgeGroup_MiddleAge"] = (input_df["Age"].between(30, 45)).astype(int)
    input_df["AgeGroup_Senior"] = (input_df["Age"].between(46, 60)).astype(int)
    input_df["AgeGroup_Elderly"] = (input_df["Age"] > 60).astype(int)
    return input_df, input_dic
      
def get_features_for_model(model_name, input_df):
  base_features = [ "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                   "HasCrCard", "IsActiveMember", "EstimatedSalary",
                   "Geography_France", "Geography_Germany", "Geography_Spain",
                   "Gender_Female","Gender_Male"]
  engineered_features = base_features + ["CLV", "TenureAgeRatio", "AgeGroup_MiddleAge", "AgeGroup_Senior", "AgeGroup_Elderly"]

  if model_name in ["KNN", "SVC", "RF", "GNB", "Decision Tree","XGB"]:
      return input_df[base_features]
  else:  # XGB models
      return input_df[engineered_features]




def predict_churn(input_df):
  updated_probabilities = {}
  for model_name, model_obj in model_dict.items():
      X = get_features_for_model(model_name, input_df)
      updated_probabilities[model_name] = model_obj.predict_proba(X)[0][1]  
  avg_probabilities = np.mean(list(updated_probabilities.values()))

  col1, col2 = st.columns(2)
  with col1:
      fig = ut.create_gauge_chart(avg_probabilities)
      st.plotly_chart(fig, use_container_width=True)
      st.write(f"Average Churn Probability: {avg_probabilities:.2%}")
  with col2:
      fig_probs = ut.create_probability_chart(updated_probabilities)
      st.plotly_chart(fig_probs, use_container_width=True)
  return avg_probabilities

svc_model = load_model('svc_model.pkl')
knn_model = load_model('kn_model.pkl')
rf_model = load_model('rf_model.pkl')
xgb_model = load_model('xgb1_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
gausian_byas_model = load_model('gnb_model.pkl')
voting_clasiifier_model = load_model('votingClassifier.pkl')
xgb_smote_model = load_model('smote.pkl')
xgb_feature_engineering_model = load_model('xgb2_model.pkl')

model_dict = {
  "KNN": knn_model,
  "RF": rf_model,
  "XGB": xgb_model,
  "GNB": gausian_byas_model,
  "Voting Classifier": voting_clasiifier_model,
  "XGB SMOTE": xgb_smote_model,
  "XGB Feature Engineering": xgb_feature_engineering_model,
  "Decision Tree": decision_tree_model
}
st.title("Customer Churn Prediction")
df = pd.read_csv('churn.csv')
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for index, row in df.iterrows()
]

st_customers = st.selectbox("Select a customer:", customers)
if st_customers:
    customerId = int(st_customers.split("-")[0])
    print("selected customer id", customerId)
    selected_customer = df.loc[df['CustomerId'] == customerId].iloc[0]
    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))
        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer['Age']))
        tenure = st.number_input("Tenure",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))
        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == "Male" else 1)

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer['Geography']))

    with col2:
        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))
        num_of_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer['NumOfProducts']))

        has_cr_card = st.checkbox("Has Credit Card",
                                  value=bool(selected_customer['HasCrCard']))
        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    input_df,input_dic = prepare_input(credit_score, age, tenure, balance,
                                        num_of_products, has_cr_card,
                                        is_active_member, estimated_salary,
                                        gender, location)

    avg_probabilities =predict_churn(input_df)
    st.markdown("### Explanation")
    explanation = explain_predictions(avg_probabilities,input_dic,selected_customer['Surname'])
    st.write(explanation)

    st.markdown("### Personalize Email")
    email = generate_email(avg_probabilities,input_dic,selected_customer['Surname'],explanation)
    st.write(email)
  
