# PROTAC-TS

**PROTAC-TS** is a PROTAC linker design method based on [ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2/) to improve cell membrane permeability.

## How to set up
### Python Dependencies
- python: 3.11
- chemtsv2: 1.1.2
- tabpfn: 2.0.3
- medchem: 2.0.5
- scikit-learn: 1.5.1
- (Optional) lightgbm, autogluon, optuna, mordred
### Example
```bash
mamba create -n protacts -c conda-forge python==3.11
mamba activate protacts
pip install chemtsv2==1.1.2 tabpfn==2.0.3 medchem==2.0.5 scikit-learn==1.5.1
```

## Quick start
Clone this repository and move into it
```bash
git clone git@github.com:ycu-iil/PROTAC-TS.git
cd PROTAC-TS
```
Construct a prediction model for cell membrane permeability and design linkers with default settings:
```bash
# 1 – feature generation
python make_feature.py -c config/setting_feature.yaml
# 2 – model training
python make_model.py -c config/setting_model.yaml
# 3 – linker design (CPU)
chemtsv2 -c config/setting_protacts.yaml
```
For GPU‑accelerated reward evaluation:
```bash
chemtsv2 -c config/setting_protacts.yaml --gpu 0 --use_gpu_only_reward
```

## Detailed workflow
### 1. Construct a prediction model for cell membrane permeability
#### 1-1. Feature generation
`make_feature.py -c config/setting_feature.yaml` calculates descriptors defined in `config/setting_feature.yaml` (e.g., Morgan fingerprint, Mordred descriptor).
#### 1-2. Model training
`make_model.py -c config/setting_model.yaml` trains the model using the settings specified in `config/setting_model.yaml`.

### 2. Design linkers
#### 2-1. Prepare a reward file for linker design (e.g., reward/reward_protacts.py)
PROTAC-TS employs ChemTSv2 as a linker generator.
Here, please prepare a reward file for PROTAC-TS according to instructions for how to define reward function in [ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2/blob/c61abbc702b914a76e076d87d416cdc67d3fd517/reward/README.md).
If you use `reward/reward_protacts.py`, you can modify the path to the permeability model within this file.

#### 2-2. Prepare a configuration file for linker design (e.g., config/setting_protacts.yaml)
Please prepare a yaml file containing the settings for PROTAC-TS. The details of these settings are described in [Setting to run PROTAC-TS](#Setting-to-run-PROTAC-TS) or [ChemTSv2](https://github.com/molecule-generator-collection/ChemTSv2/).

#### 2-3. Run PROTAC-TS
`chemtsv2 -c config/setting_protacts.yaml` launches linker design.

### 3. Post‑processing
Merge the designed linkers with ligands:
```bash
chemtsv2-add_cores_to_linker -c config/setting_protacts.yaml
```

## Setting to run PROTAC-TS
| Option | Description |
| ------------- | ------------- |
| `c_val` | An exploration parameter to balance the trade-off between exploration and exploitation. A larger value (e.g., 1.0) prioritizes exploration, and a smaller value (e.g., 0.1) prioritizes exploitation. |
| `threshold_type` | Threshold type to select how long (hours) or how many (generation_num) linker generation to perform. |
| `hours` | Time for linker generation in hours |
| `generation_num` | Number of linkers to be generated. Please note that the specified number is usually exceeded. |
| `cores` | Ligands for linker generation |
| Molecule filter | Linker filter to skip reward calculation of unfavorable generated linkers. Please refer to [Filters](#Filters) for details. |
| RNN model replacement | Users can switch RNN models used in expansion and rollout steps of PROTAC-TS. The model needs to be trained using Tensorflow. model_json specifies the JSON file containing the architecture of the RNN model, and model_weight specifies the H5 file containing its weights. token specifies the pickle file that contains the token list used when training an RNN model. |
| Reward replacement | Users can use any reward function. |

## Filters
| Module | Target | Description |
| ------------- | ------------- | ------------- |
| attachment_points_filter.py | Linker | Exclude linkers whose number of asterisks (used as ligand connection points) does not match the user-defined value | 
| linker_validation_filter.py | Linker | Exclude linkers that result in invalid molecules when attached to both ligands |
| radical_filter.py | Linker | Exclude SMILES linkers containing radical electrons |
| ring_size_filter.py | Linker | Exclude SMILES linkers containing ring substructures with user‑defined size |
| branch_filter.py | Linker | Exclude linkers whose shortest path between the two attachment points contains at least the user-defined number of branching atoms, including those in ring structures |
| linker_length_filter.py | Linker | Exclude linkers whose maximum path length between the attachment points exceeds the user-defined value |
| linker_similarity_filter.py | Linker | Excludes linkers whose maximum Tanimoto similarity to the linkers used in training the RNN-based linker generator, calculated using Morgan fingerprints, is below a user-defined value |
| structural_alert_filter.py | PROTAC | Exclude molecules that contain substructures listed under “Common Alerts” in the medchem package |
| substructure_filter.py | PROTAC | Exclude linkers that result in PROTACs containing substructures specified in SMILES or SMARTS format, which are considered synthetically challenging or chemically unstable|
| premodel_ad_filter.py | PROTAC | Exclude linkers that result in PROTACs with a maximum Tanimoto similarity below user-defined value to any of the 43 PROTACs used as training data for the prediction model for cell membrane permeability, based on Morgan fingerprints, whose radius and dimension were 2 and 2,048, respectively |

## Data
The `data` directory contains two files derived from PROTAC-DB 3.0 [1]:  
- **PROTAC_Caco-2_data.csv**: Experimental data of Caco-2 cell membrane permeability for PROTACs.  
- **protac_linker_fragments.smi**: SMILES representations of PROTAC linker fragments.

[1] Ge, J.; Li, S.; Weng, G.; Wang, H.; Fang, M.; Sun, H.; Deng, Y.; Hsieh, C.-Y.; Li, D.; Hou, T. PROTAC-DB 3.0: An Updated Database of PROTACs with Extended Pharmacokinetic Param-eters. Nucleic Acids Res 2025, 53 (D1), D1510–D1515.

## License
This package is distributed under the MIT License.

## Contact
- Yuki Murakami (w245513c@yokohama-cu.ac.jp)
- Kei Terayama (terayama@yokohama-cu.ac.jp)
