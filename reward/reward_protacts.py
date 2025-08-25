import re
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from chemtsv2.reward import Reward
from chemtsv2.misc.scaler import max_gauss

permeability_model_file = "model/pre_model/PROTAC/model_all.pkl"
with open(permeability_model_file, mode='rb') as f:
    permeability_model = pickle.load(f)

def calc_MorganCount(mol, r=2, dimension=500):
    info = {}
    _fp = AllChem.GetMorganFingerprint(mol, r, bitInfo=info)
    count_list = [0] * dimension
    for key in info:
        pos = key % dimension
        count_list[pos] += len(info[key])
    return count_list

def add_atom_index_in_wildcard(smiles: str):
    c = iter(range(1, smiles.count('*')+1))
    labeled_smiles = re.sub(r'\*', lambda _: f'[*:{next(c)}]', smiles)
    return labeled_smiles

def link_linker(conf, mol):
    smi = Chem.MolToSmiles(mol)
    mol_ = Chem.MolFromSmiles(add_atom_index_in_wildcard(smi))
    rwmol = Chem.RWMol(mol_)
    cores_mol = [Chem.MolFromSmiles(s) for s in conf['cores']]
    for m in cores_mol:
        rwmol.InsertMol(m)
    prod = Chem.molzip(rwmol)
    return prod

class Linker_permeability_reward(Reward):
    def get_objective_functions(conf):
        def Permeability(mol):
            prod = link_linker(conf, mol)
            Chem.SanitizeMol(prod)
            morganfp = calc_MorganCount(prod, r=2, dimension=500)
            X = np.array(morganfp, dtype=np.float32)
            y_pred = permeability_model.predict(X.reshape(1, -1))
            return y_pred[0]
        return [Permeability]
    
    def calc_reward_from_objective_values(values, conf):
        return max_gauss(values[0], mu=0.25, sigma=1)