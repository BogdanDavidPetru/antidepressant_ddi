from rdkit import Chem

common_mcs_file = 'common_mcs_antidepressants_smiles.txt'
common_parsed_mcs = []

with open(common_mcs_file, 'r') as fp:
    # fp.readline()
    index = 0
    for mcs in fp:
        if not common_parsed_mcs.__contains__(mcs):
            fragments1 = mcs.split('.')
            largest_smiles1 = max(fragments1, key=len)
            largest_smiles_upper = largest_smiles1.upper()
            try:
                molecule = Chem.MolFromSmiles(largest_smiles_upper)
                if molecule is not None:
                    common_parsed_mcs.append(largest_smiles_upper)
            except Exception as e:
                print(f"Could not parse {largest_smiles_upper} as Molecule")


f = open("parsed_common_mcs_antidepressants_smiles.txt", "w")

for mcs in common_parsed_mcs:
    f.write(mcs)
f.close()
# # Define the molecules
# mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
# mol2 = Chem.MolFromSmiles("CCN")  # Ethylamine
#
# # Find the Maximum Common Substructure (MCS)
# mcs_result = rdFMCS.FindMCS([mol1, mol2])
# mcs_mol = Chem.MolFromSmarts(mcs_result.smarts)
#
# # Get the atom matches for the MCS in both molecules
# match1 = mol1.GetSubstructMatch(mcs_mol)
# match2 = mol2.GetSubstructMatch(mcs_mol)
#
# # Extract the common substructure as submols
# sub_mol1 = Chem.PathToSubmol(mol1, match1)
# sub_mol2 = Chem.PathToSubmol(mol2, match2)
#
# # Step 1: Sanitize the submolecules
# Chem.SanitizeMol(sub_mol1)
# Chem.SanitizeMol(sub_mol2)
#
# # Step 2: Convert to SMILES
# # Here, we process the SMILES to handle cases with dots (".") indicating separate fragments
# smiles1 = Chem.MolToSmiles(sub_mol1)
# smiles2 = Chem.MolToSmiles(sub_mol2)
#
# # If you find fragments, you can split and choose the largest fragment:
# fragments1 = smiles1.split('.')
# fragments2 = smiles2.split('.')
#
# # Select the largest fragment (or the most relevant one)
# largest_smiles1 = max(fragments1, key=len)
# largest_smiles2 = max(fragments2, key=len)
#
# # Print the results
# print("Common substructure SMILES (mol1):", largest_smiles1)
# print("Common substructure SMILES (mol2):", largest_smiles2)
