import os
import csv

# not_researched_drugs = {12867, 12710, 12401, 13256, 13273, 13552, 13665, 13784, 13791, 13841, 13523, 4872, 4888, 8927, 11736, 12093, 12273, 12518, 12958, 13213, 13382, 13403, 13554, 13557, 13676, 372, 11540, 13420, 6077, 9266, 14185, 14651, 16021, 5766, 17056, 12543}

data_dir = 'C:/Users/david.bogdan/master/disertatie/antipsychotic-drugbank/train_test'

drug_ids = []

for db_id in os.listdir(data_dir):
    db_id = db_id[:-4]
    print(db_id)
    drug_ids.append(db_id)

interactions = []

known_DDI_file = ('C:/Users/david.bogdan/master/disertatie/example-ddi-dfi-prediction/deepddi/data/DrugBank_known_ddi'
                  '.txt')

# left_ddi_info = {}
# right_ddi_info = {}
drugs_with_interactions = set()
with open(known_DDI_file, 'r') as fp:
    fp.readline()
    for line in fp:
        # print(line)
        sptlist = line.split()  # line.strip().split('/t')

        left_drug = sptlist[0].strip()
        right_drug = sptlist[1].strip()
        interaction_type = sptlist[2].strip()
        if drug_ids.__contains__(left_drug) and drug_ids.__contains__(right_drug):
            interactions.append((left_drug, right_drug, interaction_type))
            drugs_with_interactions.add(left_drug)
            drugs_with_interactions.add(right_drug)

for interaction in interactions:
    print("Drug 1: %s, Drug 2: %s, interaction: %s" % (interaction[0], interaction[1], interaction[2]))

print("Interactions Size: ", len(interactions))

filename = "drug_interaction_data_not_valid_antipsychotics.csv"
header = ['Drug1', 'Drug2', 'Interaction']

already_parsed = set()

with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    for index, drug in enumerate(drug_ids):
        for index2, drug2 in enumerate(drug_ids):
            if index2 > index:
                interaction_found = False
                for (left_drug, right_drug, interaction_type) in interactions:
                    if (left_drug == drug and right_drug == drug2) or (left_drug == drug2 and right_drug == drug):
                        interaction_found = True
                        break
                if not already_parsed.__contains__((drug, drug2)) and not already_parsed.__contains__((drug2, drug)):
                    if interaction_found:
                        csvwriter.writerow([drug, drug2, 1])
                    else:
                        csvwriter.writerow([drug, drug2, 0])
                    already_parsed.add((drug, drug2))

print('Number of drugs taken into consideration for dataset: ', len(drug_ids))
# drugs_without_interactions = set()
#
# for drug_id in drug_ids:
#     if not drugs_with_interactions.__contains__(drug_id):
#         drugs_without_interactions.add(drug_id)

# print("Drugs without known interaction: ", len(drugs_without_interactions))
# print(drugs_without_interactions)
# print(drug_ids)