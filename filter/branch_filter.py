from rdkit import Chem

from chemtsv2.filter import Filter

class BranchFilter(Filter):
    def check(mol, config):
        try:
            smi = Chem.MolToSmiles(mol)
        except:
            return False
        if smi.count("*") != 2:
            return False
        asterisk_list = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                asterisk_list.append(atom.GetIdx())
        dist_mat = Chem.GetDistanceMatrix(mol)
        ref_len = dist_mat[asterisk_list[0]][asterisk_list[1]]
        most_short_len_atom = []
        branch_counter = 0
        for i in range(len(dist_mat[asterisk_list[0]])):
            a = dist_mat[asterisk_list[0]][i] + dist_mat[asterisk_list[1]][i]
            if a != ref_len:
                continue
            else:
                most_short_len_atom.append(i)
        if mol.GetRingInfo().NumRings() == 0:
            ring_atom = []
        else:
            ring_atom = list(mol.GetRingInfo().AtomRings()[0])
            ring_atom = [i for i in ring_atom if i not in most_short_len_atom]
        ref_atom = most_short_len_atom + ring_atom
        all_atom = [atom.GetIdx() for atom in mol.GetAtoms()]
        branch_atom = [i for i in all_atom if i not in ref_atom]
        for atom in mol.GetAtoms():
            if atom.GetIdx() not in branch_atom:
                continue
            branch_counter += 1
            #skip_counter = 0
            #for bond in atom.GetBonds():
            #    if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and atom.GetSymbol() == "O":
            #        skip_counter += 1
            #枝分かれ一個かつ炭素だったら許容
            #branch_len_list = []
            #for i in ref_atom:
            #    branch_len_list.append(dist_mat[atom.GetIdx()][i])
            #if min(branch_len_list) == 1 and atom.GetSymbol() == "C":
            #    skip_counter += 1
            #if skip_counter == 0:
            #    branch_counter += 1
        return branch_counter <= config['branch_filter']['threshold']