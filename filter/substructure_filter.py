from rdkit import Chem

from chemtsv2.filter import Filter
from chemtsv2.utils import transform_linker_to_mol

class SubstructureFilter(Filter):
    def check(mol, config):
        ref_smi = Chem.MolToSmiles(mol)
        ref_mol = Chem.MolFromSmiles(ref_smi)
        substructure_list = config["substructure_filter"]["substructure"]
        alert_counter = 0
        for sub in substructure_list:
            sub_mol_smarts = Chem.MolFromSmarts(sub)
            sub_mo_smiles = Chem.MolFromSmiles(sub)   
            if ref_mol.GetSubstructMatches(sub_mol_smarts) or ref_mol.HasSubstructMatch(sub_mo_smiles):
                alert_counter += 1
        return alert_counter == 0

class SubstructureFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return SubstructureFilter.check(mol, conf)
        return _check(mol, conf)
