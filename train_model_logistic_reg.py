import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
from sklearn.metrics import hamming_loss
import joblib

drug_interaction_df = pd.read_csv('drug_interaction_train_test_antidepressants.csv')

print("----------Initial Data Frame info-------------")
drug_interaction_df = drug_interaction_df.iloc[:, 1:]  # remove first column which represents index
print(drug_interaction_df.info())

print("----------Input Data Frame info-------------")
input_drug_interaction_df = drug_interaction_df.iloc[:, :-7]
print(input_drug_interaction_df.info())

print("----------Output Data Frame info-------------")
output_drug_interaction_df = drug_interaction_df.iloc[:, -7:]
print(output_drug_interaction_df.info())

X_train, X_test, y_train, y_test = train_test_split(input_drug_interaction_df, output_drug_interaction_df,
                                                    test_size=0.2, random_state=42)

X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.values
y_ts_arr = y_test.values

print('Input Shape', X_tr_arr.shape)
print('Input test Shape', X_test.shape)
print('Output train shape: ', y_tr_arr.shape)
print('Output test shape: ', y_ts_arr.shape)
#
# #train
#
#
multilabel_dt_model = MultiOutputClassifier(DecisionTreeClassifier(class_weight='balanced', max_depth=100, random_state=42))
#, max_iter=1000))

parameters = {
    'estimator__max_depth': [10, 20, 30, 100, None],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__class_weight': ['balanced'],
    'estimator__max_features': [None, 'sqrt', 'log2']
}

print("Starting the grid search for the best params")

grid_search = GridSearchCV(multilabel_dt_model, param_grid=parameters, scoring='f1_weighted', cv=5)
grid_search.fit(X_tr_arr, y_tr_arr)

# Best parameters
best_params = grid_search.best_params_
print("Best parameters found: ", best_params)

# Best model
best_model = grid_search.best_estimator_

# multilabel_dt_model.fit(X_tr_arr, y_tr_arr)


# pred = multilabel_regression_model.predict(X_ts_arr)

# print('Accuracy Score from sklearn: ', accuracy_score(y_ts_arr, pred))
#
# print('Hamming Loss: ', round(hamming_loss(y_ts_arr, pred), 2))

model_file_name = 'multilabel_decision_tree_model_balanced_best_weighted_f1_gridsearch.pkl'
#
# print('Finished fitting. Saving trained model to file: ', model_file_name)
#

# multilabel_regression_model = joblib.load(model_file_name)

print('Starting predictions on test data')

pred = best_model.predict(X_ts_arr)

print('Accuracy Score from sklearn: ', accuracy_score(y_ts_arr, pred))

print('Hamming Loss: ', round(hamming_loss(y_ts_arr, pred), 2))

increase_activity_conf_matrix = multilabel_confusion_matrix(y_ts_arr, pred)[0]
decrease_activity_conf_matrix = multilabel_confusion_matrix(y_ts_arr, pred)[1]
increase_effect_conf_matrix = multilabel_confusion_matrix(y_ts_arr, pred)[2]
decrease_effect_conf_matrix = multilabel_confusion_matrix(y_ts_arr, pred)[3]
increase_efficacy_conf_matrix = multilabel_confusion_matrix(y_ts_arr, pred)[4]
decrease_efficacy_conf_matrix = multilabel_confusion_matrix(y_ts_arr, pred)[5]
other_interaction_conf_matrix = multilabel_confusion_matrix(y_ts_arr, pred)[6]

print('Confusion matrix for Increase Activity outcome')
print(increase_activity_conf_matrix)
print('Confusion matrix for Decrease Activity outcome')
print(decrease_activity_conf_matrix)
print('Confusion matrix for Increase Effect outcome')
print(increase_effect_conf_matrix)
print('Confusion matrix for Decrease Effect outcome')
print(decrease_effect_conf_matrix)
print('Confusion matrix for Increase Efficacy outcome')
print(increase_efficacy_conf_matrix)
print('Confusion matrix for Decrease Efficacy outcome')
print(decrease_efficacy_conf_matrix)
print('Confusion matrix for Other Interaction outcome')
print(other_interaction_conf_matrix)

report = classification_report(y_ts_arr, pred, output_dict=False,
                               target_names=['Increase Activity Interaction', 'Decrease Activity Interaction',
                                             'Increase Effect Interaction', 'Decrease Effect Interaction',
                                             'Increase Efficacy Interaction', 'Decrease Efficacy Interaction',
                                             'Other Interaction'])

print('Report: ', report)
joblib.dump(best_model, model_file_name)


# regression_model = LogisticRegression(solver='lbfgs', max_iter=1000)
#
# regression_model.fit(X_tr_arr, y_tr_arr.ravel())
#
# pred = regression_model.predict(X_ts_arr)
#
# print('Accuracy from sk-learn: {0}'.format(regression_model.score(X_ts_arr, y_ts_arr.ravel())))

