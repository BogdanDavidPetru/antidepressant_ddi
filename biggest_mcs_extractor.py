from rdkit import Chem

common_mcs_file = 'common_mcs_antidepressants.txt'
common_mcs_dict = {}

with open(common_mcs_file, 'r') as fp:
    fp.readline()
    index = 0
    for mcs in fp:
        if not common_mcs_dict.__contains__(mcs):
            common_mcs_dict[mcs] = index
            index = index + 1

feature_vector_length = len(common_mcs_dict)
print('MCS dictionary length: ', feature_vector_length)

mcs_length_dict = {}
mcs_weighted_length_dict = {}

for mcs in common_mcs_dict.keys():
    mcs_molecule = Chem.MolFromSmarts(mcs)
    bonds = mcs_molecule.GetNumBonds()
    if mcs_length_dict.__contains__(bonds):
        mcs_length_dict[bonds].append(mcs_molecule)
    else:
        mcs_length_dict[bonds] = [mcs_molecule]

for mcs in common_mcs_dict.keys():
    mcs_molecule = Chem.MolFromSmarts(mcs)
    bonds = mcs_molecule.GetBonds()
    mcs_size = 0
    for bond in bonds:
        if bond.GetBondType() is Chem.BondType.DOUBLE:
            mcs_size += 2
        elif bond.GetBondType() is Chem.BondType.TRIPLE:
            mcs_size += 3
        elif bond.GetBondType() is Chem.BondType.AROMATIC:
            mcs_size += 4
        else:
            mcs_size += 1
    if mcs_weighted_length_dict.__contains__(mcs_size):
        mcs_weighted_length_dict[mcs_size].append(mcs_molecule)
    else:
        mcs_weighted_length_dict[mcs_size] = [mcs_molecule]

sorted_mcs_length_dict = dict(sorted(mcs_length_dict.items(), reverse=True))
sorted_mcs_weighted_length_dict = dict(sorted(mcs_weighted_length_dict.items(), reverse=True))

for key in sorted_mcs_length_dict.keys():
    nr_of_mcs = len(sorted_mcs_length_dict[key])
    print('Number of MCS with simple length %d is %d' % (key, nr_of_mcs))

for key in sorted_mcs_weighted_length_dict.keys():
    nr_of_mcs = len(sorted_mcs_weighted_length_dict[key])
    print('Number of MCS with weighted length %d is %d' % (key, nr_of_mcs))
