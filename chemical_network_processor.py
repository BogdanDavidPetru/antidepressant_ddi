from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from IPython.display import display
IPythonConsole.ipython_useSVG = True

with Chem.SDMolSupplier('./DB00027.sdf') as suppl:
    ms = [x for x in suppl if x is not None]

G = nx.Graph()
G2 = nx.Graph()


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


for m in ms:
    mol_id = m.GetProp('_Name')
    smiles = Chem.MolToSmiles(m)
    print(smiles)
    # G.add_node(mol_id, smiles=smiles)

    for atom in m.GetAtoms():
        print(atom.GetSymbol(), atom)

    for bond in m.GetBonds():
        begin_atom_id = bond.GetBeginAtomIdx()
        end_atom_id = bond.GetEndAtomIdx()
        # bond()

        # Add edges to the graph
        G.add_edge(mol_id + str(begin_atom_id), mol_id + str(end_atom_id))

    # matrix = Chem.GetAdjacencyMatrix(m, useBO=True)
    # G2 = nx.from_numpy_array(matrix)

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_size=8, font_color='black', node_size=700, node_color='skyblue',
        edge_color='gray', linewidths=0.5)

plt.show()
