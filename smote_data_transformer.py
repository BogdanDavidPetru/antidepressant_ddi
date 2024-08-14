import pandas as pd
from imblearn.over_sampling import SMOTE

data_frame = pd.read_csv('./datasets/drug_interaction_train_test_antidepressants_binary_class.csv')

print("----------Initial Data Frame info-------------")
X = data_frame.iloc[:, 3:]  # remove first column which represents index
print(X.info())

y = data_frame.iloc[:, -1:]  # extract last column which represents target
print(y.info())

smote = SMOTE(sampling_strategy=0.25)
X_sm, y_sm = smote.fit_resample(X, y)

concat = pd.concat([X_sm, y_sm], axis=1)
concat.to_csv('drug_interaction_train_test_antidepressants_binary_class_structural_smote.csv', index=False)
