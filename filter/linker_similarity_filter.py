from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

from chemtsv2.filter import Filter

linkersim_protac_linker_file = "data/protac_linker_fragments.smi"
with open(linkersim_protac_linker_file, mode='r') as f:
    linkersim_protac_linker_list = f.readlines()
linkersim_protac_linker_mol_list = [Chem.MolFromSmiles(i) for i in linkersim_protac_linker_list]
linkersim_linker_morgan_list = [rdMolDescriptors.GetMorganFingerprintAsBitVect(i, 2, nBits=2048) for i in linkersim_protac_linker_mol_list]

class LinkerSimilarityFilter(Filter):
    def check(mol, config):
        ref_morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        morgan_fp_tanimoto_list = DataStructs.BulkTanimotoSimilarity(ref_morgan_fp, linkersim_linker_morgan_list)
        return max(morgan_fp_tanimoto_list) >= config['linker_similarity_filter']['threshold']