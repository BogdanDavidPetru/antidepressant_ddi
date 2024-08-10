import shutil
import os
import pandas as pd
from rdkit import Chem

smarts = Chem.MolFromSmarts('[#7]-[#6]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1')

print(smarts)

print(Chem.MolToSmiles(smarts))
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
