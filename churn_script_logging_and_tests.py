import os
import logging
import churn_library as cls

PTH = "./data/bank_data.csv"
df = cls.import_data(PTH)
df['Churn'] = df['Attrition_Flag'].apply(
    lambda val: 0 if val == "Existing Customer" else 1)
df.drop(columns='Attrition_Flag', inplace=True)
cat_vars = [var for var in df.columns if df[var].dtype == "object"]
df_encoder = cls.encoder_helper(df, cat_vars)
X, X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
    df_encoder)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(PTH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function - Confirms that the graphics images saved by the function are in the directory.
    '''
    try:
        assert os.path.isfile('images/eda/churn_distribution.png')
        assert os.path.isfile('images/eda/customer_age_distribution.png')
        assert os.path.isfile('images/eda/marital_status_distribution.png')
        assert os.path.isfile('images/eda/total_transaction_distribution.png')
        assert os.path.isfile('images/eda/heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper - Tests that confirms the categorical columns of type object were deleted, 
    and the response parameter is working correctly.
    '''
    try:
        encoder_helper(df, cat_vars, response="Churn")
        logging.info("Testing test_encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: It appears there is something wrong in the function")
        raise err

    try:
        df_new = encoder_helper(df, cat_vars)
        for i in df_new.columns:
            assert df_new[i].dtype != "O"
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The encoder not add new columns")
        raise err

    try:
        df_new = encoder_helper(df, cat_vars)
        assert sum(df_new.columns.str.contains('_Churn')) == 5
    except AssertionError as err:
        logging.error("Testing encoder_helper: The response not work")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering - Test that confirms subset split and test size percentage considered.
    '''
    try:
        X, X_train, X_test, y_train, y_test = perform_feature_engineering(
            df_encoder)
        assert X_train.shape[0] > X_test.shape[0]
        assert y_train.shape[0] > y_test.shape[0]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The function not split the dataframe")
        raise err


def test_train_models(train_models):
    '''
    test train_models - Test that trains the model and confirms the 
    existence of .pkl files and images of results and graphs.
    '''
    try:
        X, X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
            df_encoder)
        train_models(X, X_train, X_test, y_train, y_test)
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('images/results/rf_results.png')
        assert os.path.isfile('images/results/logistic_results.png')
        assert os.path.isfile('./images/results/feature_importances.png')
        assert os.path.isfile('./images/results/roc_curve_result.png')
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: Some file of train models not found in the")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda(df))
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
