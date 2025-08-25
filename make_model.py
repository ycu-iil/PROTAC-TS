import pickle
import os
import random
import argparse
import yaml
import logging

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def setup_logger(savedir):
    """Setup logger for the script"""
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

def make_lightgbm_model(X, y, output_path, seed=0, param_tune=False, save_model=False, n_jobs=-1, feature_names=None):
    """Train and construct a LightGBM model with optional hyperparameter tuning."""
    X_all = np.array(X)
    y_all = np.array(y)
    if param_tune:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        def objective(trial):
            params = {'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                      'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                      'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                      'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                      'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                      'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                      'min_child_samples': trial.suggest_int('min_child_samples', 3, 30),
                      'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                      'max_depth': trial.suggest_int('max_depth', 3, 12)}
            model.set_params(**params)
            scores = []
            for train_idx, val_idx in cv.split(X_all):
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                y_train, y_val = y_all[train_idx], y_all[val_idx]
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                mse = np.mean((y_val - pred)**2)
                scores.append(-mse) 
            return np.mean(scores)
        model = LGBMRegressor(boosting_type='gbdt', 
                              objective='regression',
                              random_state=seed, 
                              n_jobs=n_jobs, 
                              verbose=-1) 
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=100)
        best_params = study.best_trial.params
        model.set_params(**best_params)
    else:
        model = LGBMRegressor(boosting_type='gbdt', 
                              objective='regression', 
                              random_state=seed, 
                              n_jobs=n_jobs, 
                              verbose=-1)
    X_all_train, X_all_val, y_all_train, y_all_val = train_test_split(X_all, 
                                                                      y_all, 
                                                                      random_state=10, 
                                                                      test_size=0.1)
    model.fit(X_all_train, 
              y_all_train,
              eval_metric='mean_squared_error',
              eval_set=[(X_all_val, y_all_val)])
    #save model
    if save_model:
        with open(output_path+'model_all.pkl', mode='wb') as fo:
            pickle.dump(model, fo)
        importance = model.feature_importances_
        importance = pd.DataFrame(importance, index=feature_names, columns=['Feature_importance'])
        importance.to_csv(output_path+'Feature_importance.csv')
        importance = importance.sort_values('Feature_importance', ascending=False)
        plt.barh(importance.index[:10][::-1], importance['Feature_importance'][:10][::-1])
        plt.savefig(output_path+"lightgbm_feature_importance.png", bbox_inches='tight')
        plt.close()
    return model

def result_analysis(new_ytest_list, new_ypred_list, output_path, feature_names, logger):
    """Analyze the prediction results and save the performance metrics"""
    with open(output_path+'y_pred_protac.pickle', mode='wb') as fo:
        pickle.dump(new_ypred_list, fo)
    with open(output_path+'y_PROTAC.pickle', mode='wb') as fo:
        pickle.dump(new_ytest_list, fo)
    all_min = min(min(new_ytest_list), min(new_ypred_list))
    all_max = max(max(new_ytest_list), max(new_ypred_list))
    plt.figure(figsize=(10, 8))
    plt.scatter(new_ytest_list, new_ypred_list, c='#f6aa00')
    plt.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2)
    plt.xlabel('Experimental value (log$_{10}$(μcm/s))', fontsize=22)
    plt.ylabel('Predicted value (log$_{10}$(μcm/s))', fontsize=22)
    plt.tick_params(labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(output_path+"premodel_protac_plot.png", bbox_inches='tight') 
    plt.close()
    logger.info("Prediction performance")
    logger.info(' R2：%s', format(r2_score(new_ytest_list, new_ypred_list), '.3f'))
    logger.info(' R：%s', format(np.corrcoef(new_ytest_list, new_ypred_list)[0][1], '.3f'))
    logger.info(' RMSE：%s', format(np.sqrt(mean_squared_error(new_ytest_list, new_ypred_list)), '.3f')) 
    df_result = pd.DataFrame({"R2": [r2_score(new_ytest_list, new_ypred_list)], 
                              "R": [np.corrcoef(new_ytest_list, new_ypred_list)[0][1]],
                              "RMSE": [np.sqrt(mean_squared_error(new_ytest_list, new_ypred_list))],
                              "num_features": [len(feature_names)]})
    df_result.to_csv(output_path+"result.csv", encoding="shift-jis")

def main():
    """Main function to construct prediction model"""
    parser = argparse.ArgumentParser(description="",
                                     usage=f"python {os.path.basename(__file__)} -c CONFIG_FILE")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        required=True,
                        help="path to a config file")
    with open(parser.parse_args().config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    model_type =  config['model']['type']
    if model_type == "TabPFN":
        from tabpfn import TabPFNRegressor
    elif model_type == "LightGBM":
        import optuna
        from lightgbm import LGBMRegressor
    elif model_type == "AutoGluon":
        from autogluon.tabular import TabularPredictor, TabularDataset
    df_all = pd.read_csv(config["input_file"])
    output_path = config["output_path"]
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    logger = setup_logger(output_path)
    logger.info("Start making model for PROTAC data")
    df_all = df_all.iloc[:, 1:]
    df_all = df_all.dropna(axis="columns")
    df_PROTAC = df_all[df_all["cate"] == "PROTAC"]
    X_PROTAC = df_PROTAC.select_dtypes(exclude='object') 
    X_PROTAC = X_PROTAC.drop(columns=["log10Papp"])
    y_PROTAC = df_PROTAC["log10Papp"]
    feature_names = list(X_PROTAC.columns)
    X_protac = np.array(X_PROTAC)
    y_protac = np.array(y_PROTAC)
    ypred_list, ytest_list, test_index_list = [], [], []
    kf = KFold(n_splits=len(X_protac))
    for trainval_index, test_index in kf.split(X_protac):
        test_index_list.append(test_index)
        X_trainval = np.array([X_protac[i] for i in trainval_index], dtype=np.float32)
        y_trainval = np.array([y_protac[i] for i in trainval_index], dtype=np.float32)
        X_test = np.array([X_protac[i] for i in test_index], dtype=np.float32)
        y_test = np.array([y_protac[i] for i in test_index], dtype=np.float32)
        if model_type == "TabPFN":      
            reg = TabPFNRegressor(random_state=seed,
                                  ignore_pretraining_limits=config["model"]["tabpfn"]["ignore_pretraining_limits"],
                                  device=config["model"]["tabpfn"]["device"])
            reg.fit(X_trainval, y_trainval)
            y_pred = reg.predict(X_test, 
                                 output_type="full")
        elif model_type == "LightGBM":
            make_cv = make_lightgbm_model(X_trainval, 
                                          y_trainval, 
                                          output_path,
                                          seed=seed,
                                          param_tune=config["model"]["lightgbm"]["param_tune"],
                                          save_model=False,
                                          n_jobs=config["model"]["lightgbm"]["n_jobs"],
                                          feature_names=feature_names)
            y_pred = make_cv.predict(X_test)
        elif model_type == "AutoGluon":
            train_data = TabularDataset(pd.DataFrame(X_trainval, columns=feature_names))
            train_data['log10Papp'] = y_trainval
            test_data = TabularDataset(pd.DataFrame(X_test, columns=feature_names))
            predictor = TabularPredictor(label='log10Papp').fit(train_data)
            y_pred = predictor.predict(test_data)
        ypred_list.append(y_pred)
        ytest_list.append(float(y_test))
    with open(output_path+'ypred_list_alldata.pickle', mode='wb') as fo:
        pickle.dump(ypred_list, fo)
    with open(output_path+'ytest_list_alldata.pickle', mode='wb') as fo:
        pickle.dump(ytest_list, fo)
    with open(output_path+'test_index_list_alldata.pickle', mode='wb') as fo:
        pickle.dump(test_index_list, fo)
    X = np.array(X_PROTAC, dtype=np.float32)
    y = np.array(y_PROTAC, dtype=np.float32)
    if model_type == 'TabPFN':
        model = TabPFNRegressor(random_state=seed,
                                ignore_pretraining_limits=config["model"]["tabpfn"]["ignore_pretraining_limits"],
                                device=config["model"]["tabpfn"]["device"])
        model.fit(X, y)
    elif model_type == 'LightGBM':
        model = make_lightgbm_model(X,
                                    y,
                                    output_path,
                                    seed=seed,
                                    param_tune=config["model"]["lightgbm"]["param_tune"],
                                    save_model=True,
                                    n_jobs=config["model"]["lightgbm"]["n_jobs"],
                                    feature_names=feature_names)
    elif model_type == 'AutoGluon':
        train_data = TabularDataset(pd.DataFrame(X, columns=feature_names))
        train_data['log10Papp'] = y
        predictor = TabularPredictor(label='log10Papp').fit(train_data)
        model = predictor
    with open(output_path+'model_all.pkl', mode='wb') as fo:
        pickle.dump(model, fo)
    new_ypred_list = []
    for i in ypred_list:
        if model_type == "TabPFN":
            new_ypred_list.append(float(i["mean"]))
        else:
            new_ypred_list.append(float(i))
    result_analysis(ytest_list,
                    new_ypred_list,
                    output_path,
                    feature_names,
                    logger)
    with open(output_path+'feature_names.pkl', mode='wb') as f:
        pickle.dump(feature_names, f)
    logger.info("Model type: %s", model_type)
    logger.info("Number of features: %s", len(feature_names))
    logger.info("Finished!")

if __name__ == '__main__':
    main()
