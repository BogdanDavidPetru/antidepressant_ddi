import pandas as pd
from sklearn.model_selection import train_test_split

drug_interaction_df = pd.read_csv('./datasets/drug_interaction_train_test_antidepressants_multiclass.csv') #./datasets/drug_interaction_train_test_antidepressants_subgraphs.csv


output_columns = ['Increase Activity Interaction', 'Decrease Activity Interaction', 'Increase Effect '
                                                                                    'Interaction',
                  'Decrease Effect Interaction', 'Increase Efficacy Interaction', 'Decrease Efficacy Interaction',
                  'Other Interaction']
#
unique_combinations = drug_interaction_df[output_columns].drop_duplicates().sort_values(by=output_columns).reset_index(drop=True)
#
# # Create a mapping from unique combinations to class labels
#
combination_to_class = {tuple(row): idx for idx, row in unique_combinations.iterrows()}
#
# print("Combination to class mapping:")
# print(combination_to_class)
#
# # Apply the mapping to create a new single output column
drug_interaction_df['interaction'] = drug_interaction_df[output_columns].apply(lambda row: combination_to_class[tuple(row)], axis=1)

drug_interaction_df = drug_interaction_df.drop(columns=output_columns)
# print(drug_interaction_df.head(10))



drug_interaction_df['binary_interaction'] = drug_interaction_df['interaction'].apply(lambda x: 0 if x == 0 else 1)
# print(drug_interaction_df['binary_interaction'].value_counts())
# print(drug_interaction_df.info())
# # Display the DataFrame

print(drug_interaction_df.head(10))
drug_interaction_df = drug_interaction_df.drop(columns=['interaction'])
drug_interaction_df.to_csv("drug_interaction_train_test_antidepressants_binary_class.csv", sep=',', encoding='utf-8')
# #

print("----------Initial Data Frame info-------------")
drug_interaction_df = drug_interaction_df.iloc[:, 3:]  # remove first column which represents index
print(drug_interaction_df.info())



train_df, test_df = train_test_split(drug_interaction_df, test_size=0.25, random_state=42)

train_df.to_csv('train_data_structural.csv', index=False)
test_df.to_csv('test_data_structural.csv', index=False)
