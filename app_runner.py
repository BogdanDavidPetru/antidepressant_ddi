import joblib
import numpy as np
import pandas as pd
from rdkit import Chem

from antidepressant_predictor_reasoner import MCSReasoner

drug1_structural_information_file = "./output/drug1-structural-information.txt"
drug2_structural_information_file = "./output/drug2-structural-information.txt"
explanations_file = './output/explanations.txt'


def load_mcs():
    common_mcs_list = []
    with open("used_mcs_with_index.txt", 'r') as fp:
        for mcs in fp:
            mcs_molecule = Chem.MolFromSmarts(mcs.strip())
            common_mcs_list.append(mcs_molecule)
    return common_mcs_list


def main():
    binary_model_file_name = './models/binary_classification_model_random_forest.pkl'
    binary_predictor = joblib.load(binary_model_file_name)

    common_chemical_substructures_list = load_mcs()
    second_drug_smiles = 'CN1CCC(CC1)=C1C2=CC=CC=C2C=CC2=CC=CC=C12'  # 'CN1CCC(CC1)C(=O)C1=CC=CC(NC(=O)C2=C(F)C=C(F)C=C2F)=N1'#input("Enter the first Drug smiles string: ")

    # Read the second string from standard input
    first_drug_smiles = 'NC1=NCC2N1C1=CC=CC=C1CC1=CC=CC=C21'  # 'O=C1N(CCCCNC[C@H]2CCC3=CC=CC=C3O2)S(=O)(=O)C2=CC=CC=C12'#input("Enter the second Drug smiles string: ")

    print(f'First drug: {first_drug_smiles}')
    print(f'Second drug: {second_drug_smiles}')

    drug1 = Chem.MolFromSmiles(first_drug_smiles.strip())
    drug2 = Chem.MolFromSmiles(second_drug_smiles.strip())

    prediction_input = []
    with open(drug1_structural_information_file, 'w') as file1:
        for substructure in common_chemical_substructures_list:
            if drug1.HasSubstructMatch(substructure):
                prediction_input.append(1)
                file1.write(f'Drug1 contains {Chem.MolToSmiles(substructure)}\n')
            else:
                prediction_input.append(0)
    with open(drug2_structural_information_file, 'w') as file2:
        for substructure in common_chemical_substructures_list:
            if drug2.HasSubstructMatch(substructure):
                prediction_input.append(1)
                file2.write(f'Drug2 contains {Chem.MolToSmiles(substructure)}\n')
            else:
                prediction_input.append(0)
    print(f'Initial input list {prediction_input}')
    array = np.array(prediction_input)
    reshape = array.reshape(1, -1)

    prediction = binary_predictor.predict(reshape)

    interaction_result = prediction[0] == 1
    if interaction_result:
        print(f"The 2 drugs interact")
    else:
        print(f"The 2 drugs do not interact")

    print(f"Please find explanations in {explanations_file}")
    print(f"Please find drug1 chemical structural information in {drug1_structural_information_file}")
    print(f"Please find drug2 chemical structural information in {drug2_structural_information_file}")

    mcs_smiles_list = [Chem.MolToSmiles(mol) for mol in common_chemical_substructures_list]
    mcs_reasoner = MCSReasoner(binary_predictor, mcs_smiles_list, reshape, ['binary_interaction'],
                               explanations_file)
    mcs_reasoner.analyze_decision_paths()

if __name__ == "__main__":
    main()
