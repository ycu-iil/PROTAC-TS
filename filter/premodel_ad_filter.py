from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
import pandas as pd

from chemtsv2.filter import Filter
from chemtsv2.utils import transform_linker_to_mol


premodel_csv_file = pd.read_csv("data/PROTAC_Caco-2_data.csv")
premodel_smiles_list = list(premodel_csv_file["Smiles"])
premodel_mols_list = [Chem.MolFromSmiles(i) for i in premodel_smiles_list]
premodel_morgan_list = [rdMolDescriptors.GetMorganFingerprintAsBitVect(i, 2, nBits=2048) for i in premodel_mols_list]

class PremodelADFilter(Filter):
    def check(mol, config):        
        Chem.SanitizeMol(mol)
        frag_smi = Chem.MolToSmiles(mol).split('.')
        if len(frag_smi) != 1:
            return False
        ref_morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan_fp_list = DataStructs.BulkTanimotoSimilarity(ref_morgan_fp, premodel_morgan_list)
        return max(morgan_fp_list) >= config['premodel_ad_filter']['threshold']

class PremodelADFilterForXMol(Filter):
    def check(mol, conf):
        @transform_linker_to_mol(conf)
        def _check(mol, conf):
            return PremodelADFilter.check(mol, conf)
        return _check(mol, conf)