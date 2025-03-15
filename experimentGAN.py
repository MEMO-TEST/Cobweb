from data_preprocess.bank import bank_data
from data_preprocess.credit import credit_data
from data_preprocess.census import census_data
from data_preprocess.meps import meps_data
from data_preprocess.config import bank,credit,census,meps
import numpy as np
import pandas as pd
from model.model_utils import dataPreprocess, trainModel, retrainModel, trainEnsembleModel, retrainModelWithSelection
from Cobweb.Cobweb_tabular_GAN import FairnessTest
from GAN import CTGANModel

import os
import joblib
import time

from utils.logging_handler import logger_model_test

import matplotlib.pyplot as plt

dataset_dict = {"census":census_data, "credit":credit_data, "bank":bank_data, 'meps':meps_data}
data_config = {"census":census, "credit":credit, "bank":bank, "meps": meps}

class TimingAndCount:
    def __init__(self) -> None:
        self.start_time = 0
        self.times = []
        self.end_time = 0
        self.IDS_number = 0
        self.Local_time = 0
        self.Global_time = 0

if __name__ == '__main__':
    # target config
    dataset_names = ['bank','census', 'credit', 'meps']
    model_name = 'MLP' #'MLP''RF'
    optsampler = 'oversampler'
    flag_ensemble_train = False
    flag_train = False
    flag_retest = True

    # testing config
    max_global = 100
    max_local = 1000
    select_strategy = 'BT'
    approach_name = 'Cobweb_GAN_KM' if select_strategy == "kmeans" else "Cobweb_GAN_BT"
    

    approach_name = approach_name + '_random'
    log_path = f'./result/{approach_name}/{model_name}_logging_data.csv'
    log = []
    log_index = []
    column_index = ['NID', "Time", "NPS", "Total Number", "Global Time", "Local Time"]
    
    for dataset_name in dataset_names:

        for sens_param_index in data_config[dataset_name].sensitive_param:
            log_index.append(f'{dataset_name} {sens_param_index}')
            TC = TimingAndCount() 

            print(f'{dataset_name}-{model_name}-{approach_name}-{sens_param_index} fairness test:')
            fm_logging_data = [f'{dataset_name}{sens_param_index}-{model_name}-{approach_name}-fairness test:','---------------------------------------------']
        
            result_path = f'./result/{approach_name}/{dataset_name}{sens_param_index}_{model_name}_{approach_name}.npy'
            model_path = f'model_info/{model_name}/{dataset_name}_{model_name}_model.pkl'
            projection_path = f'model_info/CTGAN/{dataset_name}_ctgan_model.pkl'
            retrain_model_path = f'retrain_model_info/{approach_name}/{dataset_name}{sens_param_index}_{model_name}_{approach_name}_retrained_model.pkl'
            ensemble_model_path = f'model_info/ensemble_model/{dataset_name}{sens_param_index}_ensemble.pkl'
            time_record_path = f'result/RQ1/times/{model_name}/{approach_name}_{dataset_name}{sens_param_index}.npy'
            plt_save_path = f'result/picture/{dataset_name}{sens_param_index}_{model_name}_{approach_name}.jpg'
            retrain_model_with_selection_path = f'retrain_model_info/{dataset_name}{sens_param_index}_{model_name}_{approach_name}_retrained_with_selection_model.pkl'
            if (not os.path.exists(f'./result/{approach_name}/')):
                os.makedirs(f'./result/{approach_name}/')

            if (not os.path.exists(ensemble_model_path)) or flag_ensemble_train:
                print('ensemble_model training!')
                trainEnsembleModel(dataset_name, sens_param_index, ensemble_model_path)
            
            X, Y, input_shape, nb_classes = dataset_dict[dataset_name]()
            # preprocess label
            data_dict = {0:[],1:[]}
            temp = []
            for i in range(len(X)):
                if Y[i][0] == 1:
                    data_dict[0].append(X[i])
                    temp.append(0)
                else:
                    data_dict[1].append(X[i])
                    temp.append(1)
            Y = np.array(temp)

            if (not os.path.exists(model_path)) or flag_train:
                print("model training!")
                trainModel(dataset_name, model_name, X, Y, model_path, optsampler)
            
            model = joblib.load(model_path)
            projection_model : CTGANModel
            projection_model = CTGANModel.load(projection_path)
            # hard-label fairness testing
            print('Fairness test runing!')
            disc_samples = np.empty((0,))
            BF = FairnessTest(data_config=data_config[dataset_name], sens_param_index = sens_param_index,
                                TimingAndCount=TC, projection_model=projection_model, select_strategy=select_strategy)
            if (not os.path.exists(result_path)) or flag_retest: 
                TC.start_time = time.time()

                disc_samples, total_generate_num, TC = BF(model=model,
                                    sens_param_index = sens_param_index,
                                    all_samples=X,
                                    data_dict=data_dict,
                                    dataset_name=dataset_name,
                                    max_global = max_global,
                                    max_local = max_local
                                    )
                disc_samples_array = np.array(disc_samples)
                disc_samples = np.array(disc_samples)
                np.save(result_path, disc_samples_array)
                TC.end_time = time.time()
                elapsed_time = TC.end_time - TC.start_time
                fm_logging_data.append(f'{approach_name} elapsed_time: {elapsed_time} s\nNum of Ids: {TC.IDS_number}  Total_samples={total_generate_num} Success Rate={TC.IDS_number/total_generate_num}')
                # np.save(time_record_path, np.array(TC.times))
                fm_logging_data.append(f'Gobal_time:{TC.Global_time} s')
                fm_logging_data.append(f'local_time:{TC.Local_time} s')
            else:
                disc_samples = np.load(result_path)
            logger_model_test.info(f"\n".join(fm_logging_data))

            log.append([TC.IDS_number, elapsed_time, TC.IDS_number/elapsed_time, total_generate_num, TC.Global_time, TC.Local_time])
        
    log = [[np.round(x, 3) for x in row] for row in log]
    df = pd.DataFrame(log, columns=column_index, index=log_index)
    df.to_csv(log_path, index=True)
                    