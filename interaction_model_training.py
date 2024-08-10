import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

drug_interaction_df_initial = pd.read_csv('./datasets/drug_interaction_train_test_antidepressants_multiclass.csv')
drug_interaction_df_initial = drug_interaction_df_initial.iloc[:, 3:]  # remove first column which represents index

drug_interaction_df = drug_interaction_df_initial[drug_interaction_df_initial['interaction'] != 0].reset_index(
    drop=True)

print("----------Initial Data Frame info-------------")
# drug_interaction_df = drug_interaction_df.iloc[:, 3:]  # remove first column which represents index
print(drug_interaction_df.info())

print("----------Input Data Frame info-------------")
input_drug_interaction_df = drug_interaction_df.iloc[:, :-1]
print(input_drug_interaction_df.info())

print("----------Output Data Frame info-------------")
output_drug_interaction_df = drug_interaction_df.iloc[:, -1:]
print(output_drug_interaction_df.info())

X_train, X_test, y_train, y_test = train_test_split(input_drug_interaction_df, output_drug_interaction_df,
                                                    test_size=0.25, random_state=42)

X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.values
y_ts_arr = y_test.values

print('Input Shape', X_tr_arr.shape)
print('Input test Shape', X_test.shape)
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

model_file_name = 'multiclass_classification_model_random_forest.pkl'

joblib.dump(rf_classifier, model_file_name)
#
print("--------------Random Forest with custom class weights-------------------")
class_weights = {
    1: 12.675,
    2: 57.227,
    3: 87.914,
    4: 1054.967,
    5: 1.478,
    6: 74.655,
    7: 4.679
}
rf_weight_classifier = RandomForestClassifier(n_estimators=200, class_weight=class_weights, max_depth=20,
                                              random_state=42)

rf_weight_classifier.fit(X_tr_arr, y_tr_arr.ravel())

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
