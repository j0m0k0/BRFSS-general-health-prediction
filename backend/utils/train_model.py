from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def delete_rows(df, column, value):
    return df[df[column] != value]

def get_highly_correlated_features(df, target_column, threshold=0.15):
    # Calculate the absolute value of the Pearson correlations with the target column
    correlations = df.corr()[target_column].abs()

    # Filter the features that have a correlation higher than the threshold
    high_correlation_features = correlations[correlations > threshold]

    # Drop the target column from the result
    high_correlation_features = high_correlation_features.drop(target_column, errors='ignore')

    return high_correlation_features.index.tolist()


def train_model_scikit(df, train_percent, eval_percent, test_percent, n_estimators, min_samples_split, max_features):
    # Convert empty strings to NaN
    df = df.replace('', np.nan)

    high_correlation_features = get_highly_correlated_features(df, 'GENHLTH', 0.15)
    # ['PHYSHLTH', 'EXERANY2', 'BPHIGH6', 'CHOLMED3', 'DIABETE4', 'HAVARTH5', 'LMTJOIN3', 'EDUCA',
    #  'EMPLOY1', 'DIFFWALK', 'ALCDAY5', '_RFHLTH', '_PHYS14D', '_MENT14D', '_TOTINDA', '_RFHYPE6',
    #  '_MICHD', '_DRDXAR3', '_LMTACT3', '_LMTWRK3', '_AGEG5YR', '_AGE80', '_AGE_G', 'WTKG3', '_BMI5',
    #  '_BMI5CAT', '_EDUCAG']
    print(high_correlation_features)
    df = df[~df['GENHLTH'].isin(['7', '9', ''])]
    print("Rows: ", df.shape[0])


    # Replace NaN values in 'GENHLTH' column with the default value '3'
    df['GENHLTH'] = df['GENHLTH'].fillna("3")

    # Ensure the values in the GENHLTH column are cleaned
    # df['GENHLTH'] = df['GENHLTH'].replace(["7", "9", ""], "3")
    
    # Convert the GENHLTH column to integer type
    df['GENHLTH'] = df['GENHLTH'].astype(int)

    # Replace NaN values in other columns with the mean of the column
    df = df.fillna(-1)

    # Split the data into features (X) and the target label (y)
    X = df[high_correlation_features]
    y = df['GENHLTH']

    # Split the data into training, evaluation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=42)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=eval_percent / (train_percent + eval_percent), random_state=42)

    # Define the resampling strategy
    over = SMOTE(sampling_strategy='auto')
    under = RandomUnderSampler(sampling_strategy='auto')

    # Apply the resampling to the training data
    X_train_resampled, y_train_resampled = over.fit_resample(X_train, y_train)
    X_train_resampled, y_train_resampled = under.fit_resample(X_train_resampled, y_train_resampled)

    # Create a Random Forest Classifier model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        random_state=42, 
        max_depth=None,
        min_samples_split=min_samples_split,
        max_features=max_features,
        )

    # model = GradientBoostingClassifier()

    # model = MLPClassifier()

    # model = DecisionTreeClassifier()

    # model = GaussianNB()

    # Train the model
    model.fit(X_train_resampled, y_train_resampled)
    pickle.dump(model, open('model.pkl', 'wb'))
    # Use the model to make predictions on the evaluation set
    y_pred_eval = model.predict(X_eval)

    # Evaluate the model's performance
    eval_accuracy = accuracy_score(y_eval, y_pred_eval)
    print(f"Model evaluation accuracy: {eval_accuracy}")
    print(classification_report(y_eval, y_pred_eval))

    # Use the model to make predictions on the test set
    y_pred_test = model.predict(X_test)

    # Evaluate the model's performance
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Model test accuracy: {test_accuracy}")
    print(classification_report(y_test, y_pred_test))

    # Return the trained model
    return model


def train_model_cross(df, n_estimators, min_samples_split, max_features):
    # Convert empty strings to NaN
    df = df.replace('', np.nan)

    high_correlation_features = get_highly_correlated_features(df, 'GENHLTH', 0.15)
    # ['PHYSHLTH', 'EXERANY2', 'BPHIGH6', 'CHOLMED3', 'DIABETE4', 'HAVARTH5', 'LMTJOIN3', 'EDUCA',
    #  'EMPLOY1', 'DIFFWALK', 'ALCDAY5', '_RFHLTH', '_PHYS14D', '_MENT14D', '_TOTINDA', '_RFHYPE6',
    #  '_MICHD', '_DRDXAR3', '_LMTACT3', '_LMTWRK3', '_AGEG5YR', '_AGE80', '_AGE_G', 'WTKG3', '_BMI5',
    #  '_BMI5CAT', '_EDUCAG']
    print(high_correlation_features)
    df = df[~df['GENHLTH'].isin(['7', '9', ''])]
    print("Rows: ", df.shape[0])


    # Replace NaN values in 'GENHLTH' column with the default value '3'
    df['GENHLTH'] = df['GENHLTH'].fillna("3")

    # Ensure the values in the GENHLTH column are cleaned
    # df['GENHLTH'] = df['GENHLTH'].replace(["7", "9", ""], "3")
    
    # Convert the GENHLTH column to integer type
    df['GENHLTH'] = df['GENHLTH'].astype(int)

    # Replace NaN values in other columns with the mean of the column
    df = df.fillna(-1)


    # Split the data into features (X) and the target label (y)
    X = df[high_correlation_features]
    y = df['GENHLTH']

    # Define the resampling strategy
    over = SMOTE(sampling_strategy='auto')
    under = RandomUnderSampler(sampling_strategy='auto')

    # Create a Random Forest Classifier model
    model = RandomForestClassifier(
        # n_estimators=n_estimators, 
        random_state=42, 
        # max_depth=None,
        # min_samples_split=min_samples_split,
        # max_features=max_features,
        )

    # Create a pipeline that first performs the resampling and then trains the model
    pipeline = make_pipeline(over, under, model)

    # Perform cross-validation on the pipeline
    cv_scores = cross_val_score(pipeline, X, y, cv=5)  # Use 5 folds, but adjust as necessary

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average cross-validation score: {np.mean(cv_scores)}")

    # Train the pipeline on the entire dataset
    pipeline.fit(X, y)

    # Save the model
    pickle.dump(pipeline, open('model.pkl', 'wb'))

    # Return the trained pipeline
    return pipeline