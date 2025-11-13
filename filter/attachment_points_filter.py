from rdkit import Chem

from chemtsv2.filter import Filter


class AttachmentPointsFilter(Filter):
    def check(mol, conf):
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return smi.count('*') == conf['attachment_points_filter']['threshold']
