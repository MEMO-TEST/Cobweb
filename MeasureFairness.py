from data_preprocess.bank import bank_data
from data_preprocess.credit import credit_data
from data_preprocess.census import census_data
from data_preprocess.meps import meps_data
from data_preprocess.config import bank,credit,census,meps
import numpy as np
from model.model_utils import dataPreprocess, trainModel, retrainModel, trainEnsembleModel, retrainModelWithSelection
from Cobweb.Cobweb_tabular import FairnessTest
from multiprocessing import Process
import os
import joblib
import time

from utils.logging_handler import logger_model_fm
from utils.fairness_metric import FairnessMeasure

dataset_dict = {"census":census_data, "credit":credit_data, "bank":bank_data, 'meps':meps_data}
data_config = {"census":census, "credit":credit, "bank":bank, "meps": meps}

def original_ID_rate(dataset_name, model_name, sens_param_index, optsampler):
    fm_logging_data = [f'{dataset_name}-{sens_param_index}-{model_name} Original model:','---------------------------------------------']
    fm = FairnessMeasure()

    model_path = f'model_info/{model_name}/{dataset_name}_{model_name}_model.pkl'
    model = joblib.load(model_path)

    IFr = fm.measure_individual_discrimination(model, data_config[dataset_name], sens_param_index, 100, 10000)
    print('individual_discrimination:', IFr)
    fm_logging_data.append(f'IFr:{IFr}')

    original_data, original_label, input_shape, nb_classes = dataset_dict[dataset_name]()
    IFo = fm.measure_individual_discrimination_original(model, data_config[dataset_name], sens_param_index, original_data)
    print('individual discrimination original:',IFo)
    fm_logging_data.append(f'IFo:{IFo}')

    logger_model_fm.info(f"\n".join(fm_logging_data))

def random_ID_rate(approach_name, dataset_name, model_name, sens_param_index, optsampler):
    ensemble_model_path = f'model_info/ensemble_model/{dataset_name}{sens_param_index}_ensemble.pkl'
    if not 'BF' in approach_name:
        IDS_path = f'result/{approach_name}/{dataset_name}{sens_param_index + 1}_{model_name}_{approach_name}.npy'
    else:
        IDS_path = f'result/{approach_name}/{dataset_name}{sens_param_index}_{model_name}_{approach_name}.npy'

    model_path = f'model_info/{model_name}/{dataset_name}_{model_name}_model.pkl'
    retrain_model_path = f'retrain_model_info/{approach_name}/{dataset_name}{sens_param_index}_retrained_model.pkl'
    if not os.path.exists(f'retrain_model_info/{approach_name}/'):
        os.makedirs(f'retrain_model_info/{approach_name}/')
    retrain_model_with_selection_path = f'retrain_model_info/{approach_name}/{dataset_name}{sens_param_index}_retrained_with_selection_model.pkl'
    
    print(f'{dataset_name}-{sens_param_index}-{model_name}-{approach_name} retrain:')
    fm_logging_data = [f'{dataset_name}-{sens_param_index}-{model_name}-{approach_name} retrain:','---------------------------------------------']
    fm = FairnessMeasure()
    disc_samples = np.load(IDS_path)
    if disc_samples.shape[1] == data_config[dataset_name].params:
        disc_samples = np.delete(disc_samples, sens_param_index, axis=1)
    retrainModel(dataset_name, model_name, approach_name,ensemble_model_path,disc_samples, retrain_model_path, sens_param_index, data_config[dataset_name], dataset_dict[dataset_name], optsampler=optsampler)
    retrain_model = joblib.load(retrain_model_path)
    IFr = fm.measure_individual_discrimination(retrain_model, data_config[dataset_name], sens_param_index, 100, 10000)
    print('individual_discrimination:', IFr)
    fm_logging_data.append(f'IFr:{IFr}')

    original_data, original_label, input_shape, nb_classes = dataset_dict[dataset_name]()
    IFo = fm.measure_individual_discrimination_original(retrain_model, data_config[dataset_name], sens_param_index, original_data)
    print('individual discrimination original:',IFo)
    fm_logging_data.append(f'IFo:{IFo}')

    logger_model_fm.info(f"\n".join(fm_logging_data))

if __name__ == '__main__':
    dataset_names = ['bank', 'census', 'credit','meps']
    approach_names = ['Cobweb_BT_random','Cobweb_GAN_BT_random', 'expga', 'LIMI']
    column_names = ['Cobweb_BT_random','Cobweb_GAN_BT_random', 'expga', 'LIMI']

    input_type = 'numpy'
    model_name = 'MLP'
    optsampler = 'oversampler'

    processes = []
    for approach_name in approach_names:
        for dataset_name in dataset_names:
            for sens_param_index in data_config[dataset_name].sensitive_param:
                process = Process(target=random_ID_rate, args=(approach_name, dataset_name, model_name, sens_param_index, optsampler))
                processes.append(process)
                process.start()

    for process in processes:
        process.join()