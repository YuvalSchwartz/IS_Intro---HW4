from sklearn.model_selection import KFold  # For K-fold cross validation
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import statistics as stat
import numpy as np
import pandas as pd
import pandas.api.types


### PART 1 ###
df = pd.read_csv("C:\\Users\\97250\\Desktop\\train_Loan.csv")  # Read the file with the given path.


### PART 2 ###
# Running over the data-frame's variable names.
for variable in df:
    # Checks whether the variable is numeric (except 'Credit_History' column which is numeric type but with categorial meaning).
    if pandas.api.types.is_numeric_dtype(df[variable]) and not variable == 'Credit_History':
        print(df[variable].describe())  # Print current variable's statistic details summary.
    else:  # variable is non-numeric.
        print(df[variable].value_counts())  # Print current variable frequency distribution.


### PART 3 ###
print(df.dtypes)  # Print the type of each variable.


### PART 4 ###
# Creates a list of object-type variables names and iterates it.
for object_variable in [x for x in df if df[x].dtype.name == 'object']:
    # Fill empty fields with the mode of the column.
    df[object_variable].fillna(stat.mode(df[object_variable]), inplace=True)
# Creates a list of non-object-type variables names and iterates it.
for numeric_variable in [x for x in df if df[x].dtype.name != 'object']:
    # This specific variable is categorial but represented as numeric in the csv.
    if numeric_variable == 'Credit_History':
        # Fill empty fields with the mode of the column.
        df[numeric_variable].fillna(stat.mode(df[numeric_variable]), inplace=True)
    else:
        # Fill empty fields with average value of the column.
        df[numeric_variable].fillna(df[numeric_variable].mean(), inplace=True)


### PART 5 ###
def binning(col, cut_points, labels=None):
    # Define min and max values:
    minval = col.min()
    maxval = col.max()

    break_points = [minval] + cut_points + [maxval]  # create list by adding min and max to cut_points

    # If no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points) + 1)

    # Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin


# Define bins as 0<=x<150, 150<=x<300, x>=300.
bins = [150, 300]
group_names = ['Low', 'Medium', 'High']  # Give each bin a name.
# Discretize the values in LoanAmount attribute.
df["LoanAmount_Bin"] = binning(df["LoanAmount"], bins, group_names)


### PART 6 ###
# Keep only the ones that are within +3 to -3 standard deviations in the column 'LoanAmount'.
df = df[(np.abs(df.LoanAmount - df.LoanAmount.mean()) <= (3 * df.LoanAmount.std()))]


### PART 7 ###
# Recalculate applicant's income and add it as a new column in the data-frame.
df = df.assign(Normalized_Income=df['ApplicantIncome'].apply(lambda x: 0.5 * np.sqrt(x)))


### PART 8 ###
df_Education = pd.get_dummies(df['Education'])  # Creates new columns for each option of value in 'Education' column
df = df.join(df_Education)  # Add the new columns to the data-frame.


### PART 9 ###
le = LabelEncoder()  # Create new encoder.
for var in [x for x in df if not pandas.api.types.is_numeric_dtype(df[x])]:  # Running over 'object' type variables.
    df[var] = le.fit_transform(df[var])  # Turn each value from object to numeric.


### PART 10 ###
df.drop(['Loan_ID', 'Education', 'ApplicantIncome', 'Not Graduate'], axis=1, inplace=True)
# Convert the data-frame to a csv file and save it in the given path.
df.to_csv(path_or_buf="C:\\Users\\97250\\Desktop\\train_Loan_updated.csv", index=False)
num_of_rows = df.shape[0]  # Number of rows in the current data-frame.
print('There are ' + str(num_of_rows) + ' rows in the data-frame.')


### PART 11 ###
# Generic function for making a classification model and accessing performance:
def classification_model(model, data_frame, predictor_vars, outcome):
    # Fit the model:
    model.fit(data_frame[predictor_vars], data_frame[outcome])

    # Make predictions on training set:
    predictions = model.predict(data_frame[predictor_vars])

    # Perform k-fold cross-validation with 10 folds
    kf = KFold(n_splits=10)
    scores = []
    for train, test in kf.split(data_frame):
        # Filter training data
        train_predictors = (data_frame[predictor_vars].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data_frame[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record accuracy from each cross-validation run
        scores.append(model.score(data_frame[predictor_vars].iloc[test, :], data_frame[outcome].iloc[test]))

    # Fit the model again so that it can be referred outside the function:
    model.fit(data_frame[predictor_vars], data_frame[outcome])
    return predictions, scores  # Return results of model for future match calculations.

outcome_var = 'Loan_Status'
predictor_vars = [x for x in df if x != 'Loan_Status']
tree_model = DecisionTreeClassifier()
tree_predictions, tree_scores = classification_model(tree_model, df, predictor_vars, outcome_var)


### PART 12 ###
# Calculate accuracy.
accuracy = metrics.accuracy_score(tree_predictions, df[outcome_var])
# Print accuracy.
print("Tree model's training accuracy: %s" % "{0:.3%}".format(accuracy))
# Print Cross-Validation score.
print("Tree model's Cross-Validation Score: %s" % "{0:.3%}".format(np.mean(tree_scores)))


### PART 13 ###
def visualize_tree(tree, features_names):
    # Create tree png using graphviz.
    # Open the generate 'tree.dot' file in notepad and copy its contents to http://webgraphviz.com/.
    dotfile = 'tree.dot'
    export_graphviz(tree, out_file=dotfile, feature_names=features_names, class_names=['No', 'Yes'])
visualize_tree(tree_model, predictor_vars)


### PART 15 ###
logistic_regression_model = LogisticRegression(max_iter=5000)  # Create new logistic regression model.
# Fit the model with the data-frame, predictor variables and the outcome variable and assign results in new variables for future calculations.
logistic_regression_predictions, logistic_regression_scores = classification_model(logistic_regression_model, df, predictor_vars, outcome_var)
# Calculate accuracy.
accuracy = metrics.accuracy_score(logistic_regression_predictions, df[outcome_var])
# Print Cross-Validation score.
print("Logistic Regression's Training accuracy: %s" % "{0:.3%}".format(accuracy))
print("Logistic Regression's Cross-Validation Score: %s" % "{0:.3%}".format(np.mean(logistic_regression_scores)))