import os
import time
from rdkit import Chem
from rdkit.Chem import rdFMCS

drug_files_directory = './drug-files/'

drug_molecule = []

for mol in os.listdir(drug_files_directory):
    full_path = os.path.join(drug_files_directory, mol)
    if os.path.isfile(full_path):
        with Chem.SDMolSupplier(full_path) as suppl:
            ms = [x for x in suppl if x is not None]
            drug_molecule.append(ms)

print(len(drug_molecule))

mcs_set = set()
start = time.time()

no_mcs_set = set()
nr = 0

for index, molecule in enumerate(drug_molecule):
    for index2, molecule2 in enumerate(drug_molecule):
        if index2 > index:
            for mol in molecule:
                for mol2 in molecule2:
                    nr += 1
                    print(f'{index, index2}')
                    mcs = rdFMCS.FindMCS([mol, mol2])
                    db_id_1 = mol.GetProp('DRUGBANK_ID')
                    db_id_2 = mol2.GetProp('DRUGBANK_ID')
                    if (mcs.numAtoms == 0 and mcs.numBonds == 0) or len(mcs.smartsString) == 0:
                        print("No common subgraph for (%s, %s)", db_id_1, db_id_2)
                        if not no_mcs_set.__contains__((db_id_2, db_id_1)):
                            no_mcs_set.add((db_id_1, db_id_2))
                    else:
                        mcs_set.add(mcs.smartsString)

print('Total combinations of MCS tried: ', nr)
print('Total unique identified MCS: ', len(mcs_set))
print('Drug pairs without identified MCS: ', len(no_mcs_set))

f = open("common_mcs_antidepressants_smiles.txt", "a")

for mcs in mcs_set:
    f.write(mcs + "\n")
f.close()

# f2 = open("drugs_without_mcs_antidepressants.txt", "a")
#
# for drug1, drug2 in no_mcs_set:
#     f2.write(drug1 + "\t" + drug2 + "\n")
# f2.close()

end = time.time()
print('Computed in: ', end-start)
