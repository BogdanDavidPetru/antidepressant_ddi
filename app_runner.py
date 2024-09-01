import sys

import joblib
import numpy as np
from rdkit import Chem

from antidepressant_predictor_reasoner import MCSReasoner

drug1_structural_information_file = "./output/drug1-structural-information.txt"
drug2_structural_information_file = "./output/drug2-structural-information.txt"
explanations_file = './output/explanations.txt'


def error_exit(cause):
    print(cause)
    sys.exit(1)


def load_mcs():
    common_mcs_molecule_list = []
    common_mcs_smarts_list = []
    with open("used_mcs_with_index.txt", 'r') as fp:
        for mcs in fp:
            mcs_strip = mcs.strip()
            mcs_molecule = Chem.MolFromSmarts(mcs_strip)
            common_mcs_molecule_list.append(mcs_molecule)
            common_mcs_smarts_list.append(mcs_strip)
    return common_mcs_molecule_list, common_mcs_smarts_list


def main():
    binary_model_file_name = './models/binary_classification_model_random_forest.pkl'
    binary_predictor = joblib.load(binary_model_file_name)
    mcs_tuple = load_mcs()
    common_chemical_substructures_list = mcs_tuple[0]
    # second_drug_smiles = 'CN1CCC(CC1)=C1C2=CC=CC=C2C=CC2=CC=CC=C12'  # 'CN1CCC(CC1)C(=O)C1=CC=CC(NC(=O)C2=C(F)C=C(F)C=C2F)=N1'#input("Enter the first Drug smiles string: ")
    second_drug_smiles = 'CN1CCC(CC1)C(=O)C1=CC=CC(NC(=O)C2=C(F)C=C(F)C=C2F)=N1'  # 'CN1CCC(CC1)C(=O)C1=CC=CC(NC(=O)C2=C(F)C=C(F)C=C2F)=N1'#input("Enter the first Drug smiles string: ")
    # # Read the second string from standard input
    # # first_drug_smiles = 'NC1=NCC2N1C1=CC=CC=C1CC1=CC=CC=C21'  # 'O=C1N(CCCCNC[C@H]2CCC3=CC=CC=C3O2)S(=O)(=O)C2=CC=CC=C12'#input("Enter the second Drug smiles string: ")
    first_drug_smiles = 'O=C1N(CCCCNC[C@H]2CCC3=CC=CC=C3O2)S(=O)(=O)C2=CC=CC=C12'  # 'O=C1N(CCCCNC[C@H]2CCC3=CC=CC=C3O2)S(=O)(=O)C2=CC=CC=C12'#input("Enter the second Drug smiles string: ")

    print(f'First drug: {first_drug_smiles}')
    print(f'Second drug: {second_drug_smiles}')

    try:
        drug1 = Chem.MolFromSmiles(first_drug_smiles.strip())
        if drug1 is None:
            error_exit("Drug 1 cannot be parsed, thus it does not represent a correct SMILES string")
    except Exception as e:
        error_exit("Drug 1 cannot be parsed, thus it does not represent a correct SMILES string")

    try:
        drug2 = Chem.MolFromSmiles(second_drug_smiles.strip())
        if drug2 is None:
            error_exit("Drug 2 cannot be parsed, thus it does not represent a correct SMILES string")
    except Exception as e:
        error_exit("Drug 2 cannot be parsed, thus it does not represent a correct SMILES string")

    prediction_input = []
    with open(drug1_structural_information_file, 'w') as file1:
        for substructure in common_chemical_substructures_list:
            if drug1.HasSubstructMatch(substructure):
                prediction_input.append(1)
                match = drug1.GetSubstructMatch(substructure)
                submol = Chem.PathToSubmol(drug1, match)
                submol_smiles = Chem.MolToSmiles(submol)
                interrupted = "interrupted" if ('.' in submol_smiles) else ""
                file1.write(f'Drug1 contains {interrupted} structure {submol_smiles}\n')
            else:
                prediction_input.append(0)
    with open(drug2_structural_information_file, 'w') as file2:
        for substructure in common_chemical_substructures_list:
            if drug2.HasSubstructMatch(substructure):
                prediction_input.append(1)
                match = drug2.GetSubstructMatch(substructure)
                submol = Chem.PathToSubmol(drug2, match)
                submol_smiles = Chem.MolToSmiles(submol)
                interrupted = "interrupted" if ('.' in submol_smiles) else ""
                file2.write(f'Drug2 contains {interrupted} structure {submol_smiles}\n')
            else:
                prediction_input.append(0)
    # print(f'Initial input list {prediction_input}')
    array = np.array(prediction_input)
    reshape = array.reshape(1, -1)

    prediction = binary_predictor.predict_proba(reshape)
    result = prediction[0]
    print(
        f"Prediction result: {result[0] * 100:1f}% chances of having no interaction, {result[1] * 100:1f}% chances of having an interaction")

    print(f"Please find explanations in {explanations_file}")
    print(f"Please find drug1 chemical structural information in {drug1_structural_information_file}")
    print(f"Please find drug2 chemical structural information in {drug2_structural_information_file}")

    # mcs_smiles_list = [Chem.MolToSmiles(mol) for mol in common_chemical_substructures_list]
    mcs_smarts_list = mcs_tuple[1]
    mcs_reasoner = MCSReasoner(binary_predictor, mcs_smarts_list, reshape, ['binary_interaction'],
                               explanations_file)
    mcs_reasoner.analyze_decision_paths()

if __name__ == "__main__":
    main()
