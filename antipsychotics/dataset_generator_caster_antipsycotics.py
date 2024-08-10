import os

data_dir = 'C:/Users/david.bogdan/master/disertatie/antipsychotic-drugbank/train'
test_dir = 'C:/Users/david.bogdan/master/disertatie/antipsychotic-drugbank/test'

drug_ids = []

for db_id in os.listdir(data_dir):
    db_id = db_id[:-4]
    print(db_id)
    drug_ids.append(db_id)

for db_id in os.listdir(test_dir):
    db_id = db_id[:-4]
    print(db_id)
    drug_ids.append(db_id)

interactions = []

known_DDI_file = 'C:/Users/david.bogdan/master/disertatie/caster-chemical-subsctructures/CASTER-master/CASTER-master/DDE/data/BIOSNAP/sup_test.csv'

# left_ddi_info = {}
# right_ddi_info = {}
drugs_with_interactions = set()
with open(known_DDI_file, 'r') as fp:
    fp.readline()
    for line in fp:
        print(line)
        sptlist = line.split(',')  # line.strip().split('/t')

        left_drug = sptlist[1].strip()
        right_drug = sptlist[3].strip()
        interaction_type = sptlist[5].strip()
        if drug_ids.__contains__(left_drug) and drug_ids.__contains__(right_drug):
            interactions.append((left_drug, right_drug, interaction_type))
            drugs_with_interactions.add(left_drug)
            drugs_with_interactions.add(right_drug)

for interaction in interactions:
    print("Drug 1: %s, Drug 2: %s, interaction: %s" % (interaction[0], interaction[1], interaction[2]))

print("Interactions Size: ", len(interactions))

drugs_without_interactions = set()

for drug_id in drug_ids:
    if not drugs_with_interactions.__contains__(drug_id):
        drugs_without_interactions.add(drug_id)

print("Drugs without known interaction: ", len(drugs_without_interactions))
