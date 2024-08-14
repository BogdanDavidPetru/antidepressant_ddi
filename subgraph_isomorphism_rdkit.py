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

def verify_exact_match(mcs_result, mol1, mol2):
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)

    # Step 3: Get the substructure match from both molecules
    match1 = mol1.GetSubstructMatch(mcs_mol)
    match2 = mol2.GetSubstructMatch(mcs_mol)

    try:
        if match1 and match2:
            # Step 4: Create new molecules based on the matching substructures
            sub_mol1 = Chem.PathToSubmol(mol1, match1)
            sub_mol2 = Chem.PathToSubmol(mol2, match2)
            sub_mol1_sanitized = Chem.SanitizeMol(sub_mol1)
            sub_mol2_sanitized = Chem.SanitizeMol(sub_mol2)

            # Step 5: Convert the substructure molecules to SMILES
            mcs_smiles1 = Chem.MolToSmiles(sub_mol1_sanitized)
            mcs_smiles2 = Chem.MolToSmiles(sub_mol2_sanitized)

            # print("MCS SMILES from Molecule 1:", mcs_smiles1)
            # print("MCS SMILES from Molecule 2:", mcs_smiles2)
            fragments1 = mcs_smiles1.split('.')
            fragments2 = mcs_smiles2.split('.')

            # Select the largest fragment (or the most relevant one)
            largest_smiles1 = max(fragments1, key=len)
            largest_smiles2 = max(fragments2, key=len)
            # Step 6: Check if the SMILES strings from both molecules are the same
            if largest_smiles1 == largest_smiles2:
                return (True, largest_smiles1)
            return (False, largest_smiles2)
    except Exception as e:
        return (False, e)
    #         print("The MCS is present in both molecules as:", mcs_smiles1)
    #     else:
    #         print("The MCS is not identical in both molecules.")
    # else:
    #     print("No common substructure found in one or both molecules.")


for index, molecule in enumerate(drug_molecule):
    for index2, molecule2 in enumerate(drug_molecule):
        if index2 > index:
            for mol in molecule:
                for mol2 in molecule2:
                    nr += 1
                    print(f'{index, index2}')
                    mcs = rdFMCS.FindMCS([mol, mol2])
                    if (mcs.numAtoms != 0 or mcs.numBonds != 0) or len(mcs.smartsString) != 0:
                        match = verify_exact_match(mcs, mol, mol2)
                        if match[0]:
                            mcs_set.add(match[1])

print('Total combinations of MCS tried: ', nr)
print('Total unique identified MCS: ', len(mcs_set))
print('Drug pairs without identified MCS: ', len(no_mcs_set))

f = open("common_mcs_antidepressants_smiles_v2.txt", "a")

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
