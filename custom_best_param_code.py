import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, confusion_matrix

best_weights = {0: 0, 1: 1}
best_minority_f1_score = 0
best_weighted_f1_score = 0
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

for weight in np.arange(0.7, 0.95 + 0.01, 0.01):
    print(f"Weight: {weight:.2f}")
    class_weights = {0: weight, 1: 1.0 - weight}
    rf_weight_classifier = RandomForestClassifier(class_weight=class_weights, random_state=42)
    rf_weight_classifier.fit(X_tr_arr, y_tr_arr.ravel())
    pred_rf = rf_weight_classifier.predict(X_ts_arr)
    report = classification_report(y_ts_arr, pred_rf, output_dict=True)

    f1_minority = report['0']['f1-score']
    f1_weighted = report['weighted avg']['f1-score']
    if f1_minority > best_minority_f1_score:
        best_weights = class_weights
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted
    elif f1_minority == best_minority_f1_score and f1_weighted > best_weighted_f1_score:
        best_weights = class_weights
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted

# print(f"Best weights: {best_weights}")

n_estimators = [50, 75, 100, 125, 150, 175, 200, 300, 400, 500]
# # depth_list: [None, 10, 20, 30],
# best_weights =  {0: 0.81, 1: 0.18999999999999995}

best_minority_f1_score = 0
best_weighted_f1_score = 0
best_estimators_nr = 0

for n_est in n_estimators:
    print(f"estimators: {n_est}")
    rf_weight_classifier = RandomForestClassifier(n_estimators=n_est, class_weight=best_weights, random_state=42)
    rf_weight_classifier.fit(X_tr_arr, y_tr_arr.ravel())
    pred_rf = rf_weight_classifier.predict(X_ts_arr)
    report = classification_report(y_ts_arr, pred_rf, output_dict=True)

    f1_minority = report['0']['f1-score']
    f1_weighted = report['weighted avg']['f1-score']
    if f1_minority > best_minority_f1_score:
        best_estimators_nr = n_est
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted
    elif f1_minority == best_minority_f1_score and f1_weighted > best_weighted_f1_score:
        best_estimators_nr = n_est
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted

print("Best estimator nr:", best_estimators_nr)

depths = [3, 5, 7, 9, 10, 20]
best_depth = 0
best_minority_f1_score = 0
best_weighted_f1_score = 0

for depth in depths:
    print(f"depth: {depth}")
    rf_weight_classifier = RandomForestClassifier(n_estimators=best_estimators_nr, class_weight=best_weights,
                                                  max_depth=depth, random_state=42)
    rf_weight_classifier.fit(X_tr_arr, y_tr_arr.ravel())
    pred_rf = rf_weight_classifier.predict(X_ts_arr)
    report = classification_report(y_ts_arr, pred_rf, output_dict=True)

    f1_minority = report['0']['f1-score']
    f1_weighted = report['weighted avg']['f1-score']
    if f1_minority > best_minority_f1_score:
        best_depth = depth
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted
    elif f1_minority == best_minority_f1_score and f1_weighted > best_weighted_f1_score:
        best_depth = depth
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted

print("Best max depth:", best_depth)

features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 'auto', 'sqrt', 'log2']
best_nr_features = 0
best_minority_f1_score = 0
best_weighted_f1_score = 0

for feature in features:
    print(f"features: {feature}")
    rf_weight_classifier = RandomForestClassifier(n_estimators=200, class_weight=best_weights, max_depth=20,
                                                  max_features=feature, random_state=42)
    rf_weight_classifier.fit(X_tr_arr, y_tr_arr.ravel())
    pred_rf = rf_weight_classifier.predict(X_ts_arr)
    report = classification_report(y_ts_arr, pred_rf, output_dict=True)

    f1_minority = report['0']['f1-score']
    f1_weighted = report['weighted avg']['f1-score']
    if f1_minority > best_minority_f1_score:
        best_nr_features = feature
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted
    elif f1_minority == best_minority_f1_score and f1_weighted > best_weighted_f1_score:
        best_nr_features = feature
        best_minority_f1_score = f1_minority
        best_weighted_f1_score = f1_weighted

print("Best max nr features:", best_nr_features)

rf_weight_classifier = RandomForestClassifier(n_estimators=200, class_weight=best_weights, max_depth=10,
                                              random_state=42)

rf_weight_classifier.fit(X_tr_arr, y_tr_arr.ravel())
# class_weights = {0: 6400, 1: 6150, 2: 6500, 3: 6700, 4: 1500, 5: 6500, 6: 4000}
# Fit the model

# best_params = grid_search.best_params_
# print("Best parameters found: ", best_params)

# Best model
# best_model = grid_search.best_estimator_

# Predict the outputs
pred_rf = rf_weight_classifier.predict(X_ts_arr)
print('Accuracy Score from sklearn: ', accuracy_score(y_ts_arr, pred_rf))

print('Hamming Loss: ', round(hamming_loss(y_ts_arr, pred_rf), 2))
conf_matrix = confusion_matrix(y_ts_arr, pred_rf)

print('Confusion matrix')
print(conf_matrix)
#
#
report = classification_report(y_ts_arr, pred_rf)
print('Report: ', report)
#
# score = report['0']['f1-score']#scorer._score_func(y_ts_arr, pred_rf, **scorer._kwargs)
# print(f"F1-score for the minority class: {score:.2f}")

# increase_activity_count = drug_interaction_df['Increase Activity Interaction'].sum()
# print(f"Number of increase activity cells with value 1: {increase_activity_count}")
#
# decrease_activity_count = drug_interaction_df['Decrease Activity Interaction'].sum()
# print(f"Number of Decrease activity cells with value 1: {decrease_activity_count}")
#
# increase_effect_count = drug_interaction_df['Increase Effect Interaction'].sum()
# print(f"Number of increase effect cells with value 1: {increase_effect_count}")
#
# decrease_effect_count = drug_interaction_df['Decrease Effect Interaction'].sum()
# print(f"Number of Decrease effect cells with value 1: {decrease_effect_count}")
#
# increase_efficacy_count = drug_interaction_df['Increase Efficacy Interaction'].sum()
# print(f"Number of Efficacy effect cells with value 1: {increase_efficacy_count}")
#
# decrease_efficacy_count = drug_interaction_df['Decrease Efficacy Interaction'].sum()
# print(f"Number of Decrease Efficacy cells with value 1: {decrease_efficacy_count}")
#
# other_count = drug_interaction_df['Other Interaction'].sum()
# print(f"Number of Other cells with value 1: {other_count}")
#
# print(drug_interaction_df.groupby(['Increase Activity Interaction', 'Decrease Activity Interaction',
#                                              'Increase Effect Interaction', 'Decrease Effect Interaction',
#                                              'Increase Efficacy Interaction', 'Decrease Efficacy Interaction',
#                                              'Other Interaction']).size().sort_values(ascending=False))


# weights = np.linspace(0.0, 0.99, 10)
#
# #Creating a dictionary grid for grid search
# param_grid = {
#     'class_weight': [{0: 0.85 - x, 1: 0.80 - x, 2: 0.90 - x, 3: 0.97-x, 4: 1.0-x, 5: x, 6: 0.97-x, 7: 0.40 - x} for x in weights]
# }

# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
# }
#
# rf_classifier = RandomForestClassifier(class_weight='balanced', random_state=42
#
# # Initialize GridSearchCV
# # grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=StratifiedKFold(), scoring='f1', n_jobs=-1, verbose=2)
#
#
#
# # gridsearch = GridSearchCV(estimator= rf_classifier,
# #                           param_grid= param_grid,
# #                           cv=StratifiedKFold(),
# #                           n_jobs=-1,
# #                           scoring='f1',
# #                           verbose=2))
