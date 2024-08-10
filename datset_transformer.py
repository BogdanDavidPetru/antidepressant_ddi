import pandas as pd

drug_interaction_df = pd.read_csv('drug_interaction_train_test_antidepressants_multiclass.csv')

drug_interaction_only_df = drug_interaction_df[drug_interaction_df['interaction'] != 0].reset_index(drop=True)

print(drug_interaction_only_df.info())

drug1_features = drug_interaction_only_df.iloc[:, 2:2892]
drug2_features = drug_interaction_only_df.iloc[:, 2893:5783]

# print(drug_interaction_only_df.iloc[1302:1303, :])
def count_common_values(row1, row2):
    return sum(row1 == row2)


# Apply the function row-wise and store the result in a new column
common_values = [count_common_values(row1, row2) for row1, row2 in zip(drug1_features.values, drug2_features.values)]

# Create a new DataFrame to store the results
result_df = pd.DataFrame({'Common Values': common_values})

print(result_df.info())

nr_substructures_drug1 = pd.DataFrame({'Drug 1 total substructures': drug1_features.sum(axis=1)})
nr_substructures_drug2 = pd.DataFrame({'Drug 2 total substructures': drug2_features.sum(axis=1)})

print(nr_substructures_drug1.info())
print(nr_substructures_drug2.info())

final_set = result_df.join(nr_substructures_drug1).join(nr_substructures_drug2)

print(pd.DataFrame({'interaction': drug_interaction_only_df['interaction']}).info())
final_set['interaction'] = drug_interaction_only_df['interaction']

print(final_set.head(10))

final_set.to_csv('simplified_antidepressants_interaction_multiclass.csv')
# output_columns = ['Increase Activity Interaction', 'Decrease Activity Interaction', 'Increase Effect '
#                                                                                     'Interaction',
#                   'Decrease Effect Interaction', 'Increase Efficacy Interaction', 'Decrease Efficacy Interaction',
#                   'Other Interaction']
#
# unique_combinations = drug_interaction_df[output_columns].drop_duplicates().sort_values(by=output_columns).reset_index(drop=True)
#
# # Create a mapping from unique combinations to class labels
#
# combination_to_class = {tuple(row): idx for idx, row in unique_combinations.iterrows()}
#
# print("Combination to class mapping:")
# print(combination_to_class)
#
# # Apply the mapping to create a new single output column
# drug_interaction_df['interaction'] = drug_interaction_df[output_columns].apply(lambda row: combination_to_class[tuple(row)], axis=1)
#
# drug_interaction_df = drug_interaction_df.drop(columns=output_columns)
# print(drug_interaction_df.head(10))

# print(drug_interaction_df['binary_interaction'].value_counts())
# print(drug_interaction_df.info())

# drug_interaction_df['binary_interaction'] = drug_interaction_df['interaction'].apply(lambda x: 0 if x == 0 else 1)
#
# # Display the DataFrame
#
# print(drug_interaction_df.head(10))
# drug_interaction_df = drug_interaction_df.drop(columns=['interaction'])
# drug_interaction_df.to_csv("drug_interaction_train_test_antidepressants_binary_class.csv", sep=',', encoding='utf-8')
#
