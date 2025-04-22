# Churn Modelling using Artificial Neural Network
This project uses a custom Artificial Neural Network (ANN) in PyTorch to build a customer churn prediction model. The goal is to classify whether a bank customer will exit or stay based on various demographic and financial features.

## Story:
In the competitive banking industry, retaining customers is crucial. This project aims to predict whether a customer will exit (churn) based on their profile and behavior. Using an artificial neural network, the model helps banks proactively identify at-risk customers, enhancing retention strategies.

## Methodology:
We approach this as a binary classification problem (Exited = 1, Stayed = 0).

The method involves:
    
    1. Preprocessing customer data (encoding + scaling)
    2. Building an Artificial Neural Network with PyTorch
    3. Training with class weighting to handle imbalance
    4. Evaluating using accuracy, confusion matrix, precision, recall, and F1-score
    5. Customizing the decision threshold to better catch churners

## The Data:
Dataset: Churn_Modelling.csv
  
  Contains 10,000 customer records from a bank.
  Each row represents a customer with:
    
    1. Demographic: Geography, Gender, Age
    2. Financial: Credit Score, Balance, Estimated Salary
    3. Activity: Tenure, Number of Products, Credit Card status, Activity status
    4. Target: Exited (0 = Stayed, 1 = Exited)

## Features Used:

    1. CreditScore:	Customer’s credit score
    2. Geography: Country (categorical)
    3. Gender: Male/Female (categorical)
    4. Age: Customer’s age
    5. Tenure: Years with the bank
    6. Balance: Account balance
    7. NumOfProducts: Number of bank products used
    8. HasCrCard: 1 if customer has a credit card
    9. IsActiveMember: 1 if the customer is an active member
    10. EstimatedSalary: Estimated yearly salary

## Libraries Used:

    1. pandas: Data manipulation
    2. torch, torch.nn: Neural network creation & training
    3. sklearn:	Preprocessing, splitting, metrics
    4. matplotlib.pyplot: Plotting a confusion matrix

## Data Exploration:

![image](https://github.com/user-attachments/assets/cb6165ca-0d9a-48a2-8b02-2530814b39a1)

## Check for Null Values:

![image](https://github.com/user-attachments/assets/e117e95b-57e1-45f9-91fd-8097dcd8f4a0)

## Model Training (Per Epoch):

![image](https://github.com/user-attachments/assets/2afd2eaf-843e-4b5d-9ffa-8416617ae6b0)
![image](https://github.com/user-attachments/assets/0276f03a-b6b1-45e8-a648-5668e2a55aae)
![image](https://github.com/user-attachments/assets/26c53089-f327-4fc1-9398-23fc84e2cacb)
![image](https://github.com/user-attachments/assets/780ca991-e1d5-4be7-8898-0996cc5b04c4)

## Final Evaluation:

Confusion Matrix Plot:

![Confusion_matrix_Ann](https://github.com/user-attachments/assets/533cc6be-bdc4-4686-92e0-bacfc25ae849)

