# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project it's for to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).


## Running Files
**How to use the Churn Library:**

<import churn_library>

Functions save in the churn_library.py file.

*Functions in Churn Library:*
<import_data(PTH)> 
Read data.

<perform_eda(df)>
Realize a Exploratory Data Analysis on df, plot and save figures in the folder images/eda.

<encoder_helper(df, category_lst, response="Churn")> 
Turn each categorial column into a new column with proportion of churn for each category.

<perform_feature_engineering(df)>
Divides df into training and test subsets, with 30% of the test size.

<train_models(X, X_train, X_test, y_train, y_test)> 
Trains and stores the best model, and saves the results and images of feature_importance and ROC Curve graphics.

## Project Structure
├── data
├── images
│   ├── eda
│   └── results
├── logs
└── models



