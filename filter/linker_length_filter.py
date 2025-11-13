from rdkit import Chem

from chemtsv2.filter import Filter

class LinkerLengthFilter(Filter):
    def check(mol, config):
        asterisk_list = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                asterisk_list.append(atom.GetIdx())
        if len(asterisk_list) != 2:
            return False
        dist_mat = Chem.GetDistanceMatrix(mol)
        if config['linker_length_filter']['upper_threshold'] == False:
            return dist_mat[asterisk_list[0]][asterisk_list[1]]-1 >= config['linker_length_filter']['lower_threshold']
        elif config['linker_length_filter']['lower_threshold'] == False:
            return dist_mat[asterisk_list[0]][asterisk_list[1]]-1 <= config['linker_length_filter']['upper_threshold']
        else:
            return dist_mat[asterisk_list[0]][asterisk_list[1]]-1 >= config['linker_length_filter']['lower_threshold'] and dist_mat[asterisk_list[0]][asterisk_list[1]]-1 <= config['linker_length_filter']['upper_threshold']