import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

train_drug_interaction_df = pd.read_csv('./datasets/train_data_structural.csv')  # './datasets/train_data_graphs.csv')
test_drug_interaction_df = pd.read_csv('./datasets/test_data_structural.csv')  # './datasets/test_data_graphs.csv')

print("----------Input Train Data Frame info-------------")
input_train_drug_interaction_df = train_drug_interaction_df.iloc[:, :-1]
print(input_train_drug_interaction_df.info())

print("----------Output Train Data Frame info-------------")
output_train_drug_interaction_df = train_drug_interaction_df.iloc[:, -1:]
print(output_train_drug_interaction_df.info())
# count_of_zeros = (output_drug_interaction_df['Interaction'] == 0).sum()
#
# print(count_of_zeros)

print("----------Input Test Data Frame info-------------")
input_test_drug_interaction_df = test_drug_interaction_df.iloc[:, :-1]
print(input_test_drug_interaction_df.info())

print("----------Output Test Data Frame info-------------")
output_test_drug_interaction_df = test_drug_interaction_df.iloc[:, -1:]
print(output_test_drug_interaction_df.info())

X_tr_arr = input_train_drug_interaction_df
X_ts_arr = input_test_drug_interaction_df
y_tr_arr = output_train_drug_interaction_df.values
y_ts_arr = output_test_drug_interaction_df.values

print('Input Shape', X_tr_arr.shape)
print('Input test Shape', X_ts_arr.shape)
print('Output train shape: ', y_tr_arr.shape)
print('Output test shape: ', y_ts_arr.shape)

print("--------------Decision Tree------------------")
dt_classifier = DecisionTreeClassifier(class_weight='balanced')

# Fit the model
dt_classifier.fit(X_tr_arr, y_tr_arr)

# Predict the outputs
pred = dt_classifier.predict(X_ts_arr)
print('Accuracy Score from sklearn: ', accuracy_score(y_ts_arr, pred))

print('Hamming Loss: ', round(hamming_loss(y_ts_arr, pred), 2))
conf_matrix = confusion_matrix(y_ts_arr, pred)

print('Confusion matrix')
print(conf_matrix)

report = classification_report(y_ts_arr, pred)
print('Report: ', report)

# model_file_name = 'binary_classification_model_decision_tree_smote_balanced.pkl'
#
# joblib.dump(dt_classifier, model_file_name)
#
print("--------------Random Forest-------------------")
rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
# Fit the model
rf_classifier.fit(X_tr_arr, y_tr_arr.ravel())

# Predict the outputs
pred_rf = rf_classifier.predict(X_ts_arr)
print('Accuracy Score from sklearn: ', accuracy_score(y_ts_arr, pred_rf))

print('Hamming Loss: ', round(hamming_loss(y_ts_arr, pred_rf), 2))
conf_matrix = confusion_matrix(y_ts_arr, pred_rf)

print('Confusion matrix')
print(conf_matrix)

report = classification_report(y_ts_arr, pred_rf)
print('Report: ', report)

print("--------------Random Forest with custom class weights-------------------")

weights = np.linspace(0.70, 0.95, 30)

# Creating a dictionary grid for grid search
param_grid = {
    'class_weight': [{0: x, 1: 1.0 - x} for x in weights],
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_classifier = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=StratifiedKFold(), scoring='f1', n_jobs=-1, verbose=2)


gridsearch = GridSearchCV(estimator=rf_classifier,
                          param_grid=param_grid,
                          cv=StratifiedKFold(),
                          n_jobs=-1,
                          scoring='f1_weighted',
                          verbose=2)

gridsearch.fit(X_tr_arr, y_tr_arr.ravel())

# Get the best parameters and the best score
print("Best Parameters:", gridsearch.best_params_)
print("Best F1-Weighted Score:", gridsearch.best_score_)

# Evaluate on the test set using the best estimator
best_clf = gridsearch.best_estimator_
y_pred = best_clf.predict(X_ts_arr)

print('Accuracy Score from sklearn: ', accuracy_score(y_ts_arr, y_pred))

print('Hamming Loss: ', round(hamming_loss(y_ts_arr, y_pred), 2))
conf_matrix = confusion_matrix(y_ts_arr, y_pred)

print('Confusion matrix')
print(conf_matrix)

report = classification_report(y_ts_arr, y_pred)

model_file_name = 'binary_classification_model_random_forest__graphs_balanced.pkl'

joblib.dump(best_clf, model_file_name)
