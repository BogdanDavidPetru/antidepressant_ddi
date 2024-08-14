import shutil
import os
import pandas as pd
from rdkit import Chem

# smarts = Chem.MolFromSmarts('[#7]-[#6]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1')
#
# print(smarts)
#
# print(Chem.MolToSmiles(smarts))

smiles1 = Chem.MolFromSmiles('NC1=NCC2N1C1=CC=CC=C1CC1=CC=CC=C21')

smiles2 = Chem.MolFromSmiles('ccc1[nH]cc(CCN(C)C)c1cc'.upper())#ccccc(cN(c)c)cc1:c:c:c:c:c:1

print(smiles2)
print(Chem.MolToSmiles(smiles2))
# smarts = Chem.MolToSmarts(smiles2)
#
# print(smiles1.HasSubstructMatch(smiles2))
# print(smiles1.HasSubstructMatch(Chem.MolFromSmarts(smarts)))
# print(Chem.MolToSmiles(smiles2))
# print(Chem.MolToSmarts(smiles2))
#
# from_smarts = Chem.MolFromSmarts(
#     '[#7](:[#6]1:[#6]-,:[#6]-,:[#6](-,:[#6]-,:[#6]-,:1)-[#7]-[#6]):[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1')
#
# smiles = Chem.MolToSmiles(from_smarts)
# print(smiles)
#
# from_smiles = Chem.MolFromSmiles(smiles)
#
# print(Chem.MolToSmarts(from_smiles))
# drug_files_directory = './drug-files/'
# drug_file_set = set()
# for drug_file in os.listdir(drug_files_directory):
#     if drug_file.endswith('.sdf'):
#         drug_file_set.add(drug_file[:-4])
#
# drug_name_file = './info/drug_agent_list_drugbank.txt'
# #
# #
# #
# # all_drug_files_directory = ('C:/Users/david.bogdan/master/disertatie/example-ddi-dfi-prediction/deepddi/data/DrugBank5'
# #                             '.0_Approved_drugs')
# #
# drug_names = []
# with open(drug_name_file, 'r') as fp:
#     for drug in fp:
#         strip = drug.strip()
#         if drug_names.__contains__(strip):
#             print(strip)
#         drug_names.append(strip)
#
# known_ddi_file = ('C:/Users/david.bogdan/master/disertatie/oregano/oregano-master/oregano-master/Integration'
#                   '/Integration V2.1/DrugBank/interaction_drugs_drugbank.tsv')
#
# DDI = pd.read_csv(known_ddi_file, sep="\t", engine="python", names=["subject", "predicate", "object"])
#
# DRUGS = {}
# n = 0
# identified_ddi = set()
# for index in range(len(DDI["subject"])):
#     if n % 100000 == 0:
#         print(n, " / ", len(DDI["subject"]))
#     if DDI["predicate"][index] == "is":
#         if drug_names.__contains__(DDI["object"][index]):
#             DRUGS[DDI["object"][index]] = DDI["subject"][index]
#     n += 1
#
# for drug_name, drug_id in DRUGS.items():
#     if not drug_file_set.__contains__(drug_id):
#         print(drug_name)
#
# copied_drug_names = set()
#
# for drug_file in os.listdir(all_drug_files_directory):
#     db_id = drug_file[:-4]
#     if db_id in DRUGS.values():
#         drug_name = {i for i in DRUGS if DRUGS[i] == db_id}
#         copied_drug_names.add(list(drug_name)[0])
#         full_drug_path = os.path.join(all_drug_files_directory, drug_file)
#         shutil.copy(full_drug_path, drug_files_directory)
#
# print('Copied: ', len(copied_drug_names))
#
# not_copied_drug_names = set()
# for drug in drug_names:
#     if not copied_drug_names.__contains__(drug):
#         not_copied_drug_names.add(drug)
#
# print('Not Copied: ', len(not_copied_drug_names))
# print(not_copied_drug_names)
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

