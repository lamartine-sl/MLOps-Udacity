# library doc string
'''
This library it's for to identify credit card customers that are most likely to churn

Author: Lamartine Santana
Date: July 13, 2021
'''

# import libraries
from feature_engine.encoding import MeanEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


sns.set()
plt.rcParams["figure.figsize"] = 20, 10
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', None)

# Constants
keep_cols = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']


def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    '''

    data = pd.read_csv(path)
    return data


def perform_eda(data):
    '''
    perform eda on data and save figures to images folder
    input:
            data: pandas dataframe

    output:
            None
    '''
    sns.barplot(
        x=data['Churn'].value_counts().index,
        y=data['Churn'].value_counts().values,
        data=data
    )
    plt.savefig('images/eda/churn_distribution.png')
    plt.clf()

    sns.distplot(
        data['Customer_Age'],
        hist=True,
        kde_kws={'color': 'r', 'lw': 3, 'label': 'KDE'}
    )
    plt.savefig('images/eda/customer_age_distribution.png')
    plt.clf()

    sns.barplot(
        x=data['Marital_Status'].value_counts().index,
        y=data['Marital_Status'].value_counts('normalize').values,
        data=data
    )
    plt.savefig('images/eda/marital_status_distribution.png')
    plt.clf()

    sns.distplot(
        data['Total_Trans_Ct'],
        hist=True,
        kde_kws={'color': 'r', 'lw': 3, 'label': 'KDE'}
    )
    plt.savefig('images/eda/total_transaction_distribution.png')
    plt.clf()

    sns.heatmap(
        data.corr(),
        annot=False,
        linewidths=2
    )
    plt.savefig('images/eda/heatmap.png')
    plt.clf()


def encoder_helper(data, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that 
            could be used for naming variables or index y column]

    output:
            data: pandas dataframe with new columns for
    '''
    encoder = MeanEncoder(variables=category_lst)
    encoder.fit(data[category_lst], data['Churn'])
    data_encoder = encoder.transform(data[category_lst])

    data_encoder.columns = [
        str(col) + '_' + response for col in data_encoder.columns]

    data_new = pd.concat([data, data_encoder], axis=1)

    data_new.drop(columns=category_lst, inplace=True)

    return data_new


def perform_feature_engineering(data):
    '''
    input:
              data: pandas dataframe
              response: string of response name [optional argument that could be used 
              for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    X = pd.DataFrame()
    X[keep_cols] = data[keep_cols]

    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X, X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest Results
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/rf_results.png')
    plt.clf()

    # Logistic Regression Results
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/logistic_results.png')
    plt.clf()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    # Create Plot
    plt.figure(figsize=(20, 15))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + 'feature_importances.png')
    plt.clf()


def train_models(X, X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Train Models
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Load models
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(rfc_model, X, './images/results/')

    # Plot and save ROC Curve
    lrc_plot = plot_roc_curve(lr_model, X_test, y_test)
    plt.figure(figsize=(20, 10))
    ax = plt.gca()
    roc_rf_plot = plot_roc_curve(rfc_model, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.clf()
