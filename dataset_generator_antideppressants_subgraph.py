import pandas as pd
import csv
import os
from rdkit import Chem

data_dir = './drug-files/'

drug_molecule_ids_dict = {}

for db_id in os.listdir(data_dir):
    molecule_file = os.path.join(data_dir, db_id)
    if os.path.isfile(molecule_file):
        with Chem.SDMolSupplier(molecule_file) as suppl:
            molecule = [x for x in suppl if x is not None]
            drug_molecule_ids_dict[db_id[:-4]] = molecule[0]

common_mcs_file = 'parsed_common_mcs_antidepressants_smiles.txt'
common_mcs_dict = {}

with open(common_mcs_file, 'r') as fp:
    index = 0
    for mcs in fp:
        if not common_mcs_dict.__contains__(mcs):
            common_mcs_dict[mcs] = index
            index += 1

feature_vector_length = len(common_mcs_dict)
print(common_mcs_dict)
print('MCS dictionary length: ', feature_vector_length)
print('Index: ', index)

dataset_header = []

for feature_number in range(feature_vector_length):
    dataset_header.insert(feature_number, 'feature_' + str(feature_number) + '_drug1')
    dataset_header.insert(feature_number + feature_vector_length, 'feature_' + str(feature_number) + '_drug2')

dataset_header.insert(2 * feature_vector_length + 1, 'Interaction')

print(dataset_header)
print('Length of data header: ', len(dataset_header))

data = pd.DataFrame(columns=dataset_header)

feature_vector_list = []

source_file = open('./datasets/drug_interaction_data_antidepressants.csv', encoding="utf8", newline='')
drug_interaction_file_reader = csv.reader(source_file, delimiter=",", quotechar='"')
n = 0

for row_nr, row in enumerate(drug_interaction_file_reader):
    if n % 1000 == 0:
        print('Parsed %d interactions.' % n)
    if row_nr != 0:
        drug1 = row[0]
        drug2 = row[1]
        increase_activity_interaction = int(row[2])
        decrease_activity_interaction = int(row[3])
        increase_effect_interaction = int(row[4])
        decrease_effect_interaction = int(row[5])
        increase_efficacy_interaction = int(row[6])
        decrease_efficacy_interaction = int(row[7])
        other_interaction = int(row[8])
        common_mcs = common_mcs_dict.keys()
        if drug_molecule_ids_dict.__contains__(drug1) and drug_molecule_ids_dict.__contains__(drug2):
            feature_vector = {}
            molecule_drug1 = drug_molecule_ids_dict.get(drug1)
            molecule_drug2 = drug_molecule_ids_dict.get(drug2)
            for mcs in common_mcs:
                mcs_index = common_mcs_dict.get(mcs)
                feature_column_name = dataset_header[mcs_index]
                feature2_column_name = dataset_header[mcs_index + feature_vector_length]
                mcs_molecule = Chem.MolFromSmiles(mcs)
                if molecule_drug1.HasSubstructMatch(mcs_molecule):
                    feature_vector[feature_column_name] = 1
                else:
                    feature_vector[feature_column_name] = 0
                if molecule_drug2.HasSubstructMatch(mcs_molecule):
                    feature_vector[feature2_column_name] = 1
                else:
                    feature_vector[feature2_column_name] = 0
            if (increase_activity_interaction + decrease_activity_interaction + increase_effect_interaction + decrease_effect_interaction + increase_efficacy_interaction + decrease_efficacy_interaction + other_interaction) > 0:
                feature_vector['Interaction'] = 1
            else:
                feature_vector['Interaction'] = 0
            data = pd.concat([data, pd.DataFrame([feature_vector])], ignore_index=True)
        # else:
        #     print('Interaction skipped cause sdf file not found: ', drug1, drug2)
    n += 1

train_test_file_name = "drug_interaction_train_test_antidepressants_subgraphs.csv"
data.to_csv(train_test_file_name, sep=',', encoding='utf-8')
print(data.head(10))
