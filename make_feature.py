import os
import pickle
import math
import argparse
import logging

import yaml
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def setup_logger(savedir):
    """Setup logger for the script."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fh = logging.FileHandler(filename=os.path.join(savedir, 'run.log'), mode='w')
    ch.setLevel(logging.INFO)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def calc_MorganCount(mol, r=2, dimension=2048):
    """Calculate count-based Morgan fingerprint"""
    info = {}
    _fp = AllChem.GetMorganFingerprint(mol, r, bitInfo=info)
    count_list = [0] * dimension
    for key in info:
        pos = key % dimension
        count_list[pos] += len(info[key])
    return count_list

def smi_to_samping(smi_list, sampling_num=3):
    """Convert SMILES to sampled molecules"""
    result_mols_list = []
    for smi in smi_list:
        mols_list = []
    for i in range(sampling_num):
        mol = Chem.MolFromSmiles(smi)
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=50*i)
        AllChem.MMFFOptimizeMolecule(mol)
        mols_list.append(mol)
    result_mols_list.append(mols_list)
    return result_mols_list

def calc_mordred_multi(result_mols_list):
    """Calculate Mordred descriptors"""
    calc = Calculator(descriptors, ignore_3D=False)
    mordred_feature = calc.pandas(result_mols_list[0])
    pre_fea_name = list(mordred_feature.columns)
    df_mordred = pd.DataFrame(columns=pre_fea_name)
    for mols_list in result_mols_list:
        mordred_feature = calc.pandas(mols_list)
        pre_fea_name = list(mordred_feature.columns)
        mordred_feature = mordred_feature.select_dtypes(exclude='object')
        mordred_feature = mordred_feature.mean()
        df_pre_mordred = pd.DataFrame(mordred_feature).T
        df_mordred = pd.concat([df_mordred, df_pre_mordred])
    df_mordred.reset_index(drop=True, inplace=True)
    return df_mordred

def main():
    """Main function to generate features"""
    parser = argparse.ArgumentParser(description="",
                                     usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE")
    parser.add_argument("-c",
                        "--config",
                        type=str, 
                        required=True,
                        help="path to a config file")
    with open(parser.parse_args().config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    fea_type = config["feature"]["type"]
    if fea_type == "mordred":
        from mordred import Calculator, descriptors
    data_protac = pd.read_csv(config["input_file"])
    target_column = config["target_column"]
    output_path = config["output_path"]
    d_num = config["feature"]["morganfp"]["d_num"] #2048 or 1024
    count_base = config["feature"]["morganfp"]["use_count_base"] #True or False
    d_type = config["feature"]["mordred"]["d_type"] #2D or 2D3D
    samp_num = config["feature"]["mordred"]["sampling_num"] #3 or 5
    delete_less_data = config["option"]["remove_censored_data"] #True or False
    delete_smiles_list = config["option"]["exclude_smiles"]
    
    try:
      os.makedirs(output_path)
    except FileExistsError:
      pass
    logger = setup_logger(output_path)
    logger.info("Start making features for PROTAC data")
    if delete_less_data:
      data_protac = data_protac[~data_protac[target_column].str.contains("<", na=False)]
      data_protac = data_protac.reset_index()
    if delete_smiles_list is not None:
        pre_smiles_protac = list(data_protac["Smiles"])
        pre_mols_protac = [Chem.MolFromSmiles(i) for i in pre_smiles_protac]
        for delete_smiles in delete_smiles_list:
            delete_mol = Chem.MolFromSmiles(delete_smiles)
            delet_index = [i for i, mol in enumerate(pre_mols_protac) if mol.HasSubstructMatch(delete_mol)]
            data_protac = data_protac.drop(delet_index)
    caco = list(data_protac[target_column])
    # nm/s to log１０（μcm/s）
    papp_protac = []
    for i in caco:
        i = str(i)
        if "<" in i:
            i = i.replace("<", "")
            i = float(i) / 2
        i = float(i)
        i = i / 10
        i = math.log10(i)
        papp_protac.append(i)
    df_protac_papp = pd.DataFrame({"cate": ["PROTAC" for i in range(len(papp_protac))],
                                  "log10Papp": papp_protac
                                 })
    smiles_protac = list(data_protac["Smiles"])
    if fea_type == "morganfp":
        if count_base:
            morgan_protac = [calc_MorganCount(Chem.MolFromSmiles(smi), 2, d_num) for smi in smiles_protac]
        else:
            morgan_protac = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, d_num) for smi in smiles_protac]
        feature_names = ["FP"+str(i) for i in range(d_num)]
        morgan_protac = pd.DataFrame(np.vstack(morgan_protac), columns=feature_names)
        df_morgan_protac = pd.concat([morgan_protac, df_protac_papp], axis=1)
        df_morgan_protac.to_csv(output_path+"fea_morganfp.csv")
        with open(output_path+'fea_morgan_list.pickle', mode='wb') as fo:
            pickle.dump(feature_names, fo)
    elif fea_type == "mordred":
        if d_type == "2D":
            calc = Calculator(descriptors, ignore_3D=True)
            mols_protac = [Chem.MolFromSmiles(smi) for smi in smiles_protac]
            mordred_feature_protac = calc.pandas(mols_protac)
            mordred_feature_protac = mordred_feature_protac.select_dtypes(exclude='object')
            df_mordred_protac = pd.DataFrame(mordred_feature_protac)
            df_mordred_protac = pd.concat([df_mordred_protac, df_protac_papp], axis=1)
            df_mordred_protac.to_csv(output_path+"fea_mordred_2d.csv")
        elif d_type == "2D3D":
            mol_protac_list = smi_to_samping(smiles_protac, sampling_num=samp_num)
            df_mordred_protac = calc_mordred_multi(mol_protac_list)
            df_mordred_protac = pd.concat([df_mordred_protac, df_protac_papp], axis=1)
            df_mordred_protac.to_csv(output_path+"fea_mordred_2d3d.csv")
        feature_names = df_mordred_protac.columns.tolist()
    else:
        raise ValueError("Feature type must be 'morganfp' or 'mordred'.")
    logger.info("Feature type: %s", fea_type)
    if fea_type == "morganfp":
        logger.info("MorganFP dimension: %d", d_num)
    elif fea_type == "mordred":
        if d_type == "2D":
            logger.info("Mordred 2D dimension: %d", len(feature_names)-2)
        elif d_type == "2D3D":
            logger.info("Mordred 2D3D dimension: %d", len(feature_names)-2)
    logger.info("Number of data (PROTAC)：%d", len(smiles_protac))
    logger.info('Finished!')

if __name__ == '__main__':
    main()