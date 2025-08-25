from rdkit import Chem

from chemtsv2.filter import Filter
from chemtsv2.utils import add_atom_index_in_wildcard


class LinkerValidationFilter(Filter):
    def check(mol, conf):
        try:
            # Check if the molecule has a valid SMILES representation
            smi = Chem.MolToSmiles(mol)
        except:
            return False
        mol_ = Chem.MolFromSmiles(add_atom_index_in_wildcard(smi))
        rwmol = Chem.RWMol(mol_)
        cores_mol = [Chem.MolFromSmiles(s) for s in conf['cores']]
        for m in cores_mol:
            rwmol.InsertMol(m)
        try:
            prod = Chem.molzip(rwmol)
            Chem.SanitizeMol(prod)
        except:
            return False
        return True
