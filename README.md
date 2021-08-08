# Credit-risk-assessment
**Data**
The dataset we’re using can be found on Kaggle and it contains data for 32,581 borrowers and 11 variables related to each borrower
**Goal**
Goal of this project to assess credit risk by two classification models XGBoost and Logit and to compare with each other, to see which model can be used to determine the defaulters more precisely.
**Abstract**
Data is downloaded from Kaggle website and the data primarily consist of a banks book of defaulters and non defaulters and each loan borrowers income, and their credit history, employment and their home status. Data is preprocessed using pandas and numpy libraries of python i.e., outliers are removed, dummy variables created for some categorical variables. The data have 78 % of non-defaulters and 22% defaulters, train-test split is performed in 80-20 ratio and further logit and XGBoost models are fitted on the train and test data by using Sci-kit learn package in python. Finally, by observing Roc curve and also classification matrix ( precision and F1 score) XGBoost model is suggested for using in credit risk assessment. 
**Packages Used**
The required packages for preprocessing and exploratory analysis are Pandas, Numpy, matplotlib, seaborn. Packages used for modelling and classification matrix and Roc are Scikit-learn, xgboost packages. 
**Conclusion**
•	XGBoost have better ROC-AUC, F1, Precision, Recall scores than Logit
•	Person_home_ownership is most important feature in predicting the defaulters and non-defaulters. 
•	Interestingly, Income is the second most important feature in predicting the loan_status variable. 

**References:**
Beshr, S. (2021, July 21). A machine learning approach to credit risk assessment. Medium. https://towardsdatascience.com/a-machine-learning-approach-to-credit-risk-assessment-ba8eda1cd11f. 
Mumtaz, A. (2021, January 4). How to develop a credit risk model and scorecard. Medium. https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03. 
Bananzi, P. (2020, December 4). Credit risk modelling in python. Medium. https://medium.com/analytics-vidhya/credit-risk-modelling-in-python-3ab4b00f6505. 



