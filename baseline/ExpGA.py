from __future__ import division
import os, sys
# sys.path.insert(0, '../')  # the code for fair classification is in this directory
# sys.path.append('../')
import numpy as np
import random
import time
from data_preprocess.config import census, credit, bank, meps
import joblib
# from sklearn.externals import joblib
import lime
import shap
# from keras.models import load_model
import signal
from lime.lime_tabular import LimeTabularExplainer
from baseline.Genetic_Algorithm import GA
# import keras.backend as K
from data_preprocess.census import census_data
from data_preprocess.credit import credit_data
from data_preprocess.bank import bank_data
from data_preprocess.meps import meps_data
# from tensorflow.python.platform import flags
from scipy.spatial.distance import  cdist
import pandas as pd

import copy

# FLAGS = flags.FLAGS

global_disc_inputs = set()
global_disc_inputs_list = []
local_disc_inputs = set()
local_disc_inputs_list = []
tot_inputs = set()
location = np.zeros(40)

"""
census: 9,1,8 for gender, age, race
credit: 9,13 for gender,age
bank: 1 for age
meps: 3 for gender
"""
data_config = {"bank": bank, "census": census, "credit": credit, 'meps': meps}
dataset_names = ['bank','census','credit','meps']
# dataset_names = ['meps']
data = {"census": census_data, "credit": credit_data, "bank": bank_data, 'meps': meps_data}

model_name = 'MLP' #'MLP''SVM''RF'

dataset = None
sensitive_param = None
total_local_count = None
config = None  
threshold_l = None
threshold = None
model = None


def ConstructExplainer(train_vectors, feature_names, class_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(train_vectors, feature_names=feature_names,
                                                       class_names=class_names, discretize_continuous=False)
    return explainer


def Shap_value(model, test_vectors):
    # background = np.zeros((1,13))
    background = shap.kmeans(test_vectors, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(test_vectors)
    return shap_values


def Searchseed(model, feature_names, sens_name, explainer, train_vectors, num, X_ori):
    seed = []
    # print(train_vectors.shape)
    for x in train_vectors:
        tot_inputs.add(tuple(x))
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num)
        explain_labels = exp.available_labels()
        exp_result = exp.as_list(label=explain_labels[0])
        rank = []
        for j in range(len(exp_result)):
            rank.append(exp_result[j][0])
        loc = rank.index(sens_name)
        # print('loc:',loc)
        location[loc] = location[loc]+1
        if loc < threshold_l:
            seed.append(x)
            imp = []
            for item in feature_names:
                pos = rank.index(item)
                imp.append(exp_result[pos][1])
        if len(seed) >= 100:
            return seed
    return seed


def Searchseed_Shap(feature_names,sens_name, shap_values, train_vectors):
    seed = []
    for i in range(len(shap_values[0])):
        sample = shap_values[0][i]
        sorted_shapValue = []
        for j in range(len(sample)):
            temp = []
            temp.append(feature_names[j])
            temp.append(sample[j])
            sorted_shapValue.append(temp)
        sorted_shapValue.sort(key=lambda x: abs(x[1]), reverse=True)
        exp_result = sorted_shapValue
        print('shap_value:' + str(exp_result))
        rank = []
        for k in range(len(exp_result)):
            rank.append(exp_result[k][0])
        loc = rank.index(sens_name)
        if loc < 10:
            seed.append(train_vectors[i])
        if len(seed) > 10:
            return seed
    return seed


class Global_Discovery(object):
    def __init__(self, stepsize=1):
        self.stepsize = stepsize

    def __call__(self, iteration,params,input_bounds,sensitive_param):
        s = self.stepsize
        samples = []
        while len(samples) < iteration:
            x = np.zeros(params)
            for i in range(params):
                random.seed(time.time())
                x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
            x[sensitive_param - 1] = 0
            samples.append(x)
        return samples


def evaluate_global(inp):
    inp0 = [int(i) for i in inp]
    inp1 = [int(i) for i in inp]

    value = random.randint(config.input_bounds[sensitive_param - 1][0], config.input_bounds[sensitive_param - 1][1])
    inp1[sensitive_param - 1] = value

    inp0 = np.asarray(inp0)
    inp0 = np.reshape(inp0, (1, -1))

    inp1 = np.asarray(inp1)
    inp1 = np.reshape(inp1, (1, -1))

    out0 = model.predict(inp0)
    out1 = model.predict(inp1)
    print(out0,out1)

    tot_inputs.add(tuple(np.delete(inp0[0], sensitive_param - 1)))

    if (abs(out0 - out1) > threshold and tuple(np.delete(inp0[0], sensitive_param - 1)) not in global_disc_inputs):
        global_disc_inputs.add(tuple(np.delete(inp0[0], sensitive_param - 1)))
        global_disc_inputs_list.append(inp0.tolist()[0])
    return abs(out1 + out0)


def evaluate_local(inp):
    inp0 = [int(i) for i in inp]
    inp0 = np.array(inp0)

    inp0 = np.reshape(inp0, (1, -1))
    out0 = model.predict(inp0)

    tot_inputs.add(tuple(np.delete(inp0[0], sensitive_param-1)))
    global total_local_count
    total_local_count += 1
    for val in range(config.input_bounds[sensitive_param-1][0], config.input_bounds[sensitive_param-1][1]+1):
        if val != inp[sensitive_param-1]:
            inp1 = [int(i) for i in inp]
            inp1[sensitive_param - 1] = val

            inp1 = np.array(inp1)
            inp1 = np.reshape(inp1, (1, -1))

            out1 = model.predict(inp1)

            # pre0 = model.predict_proba(inp0)[0]
            # pre1 = model.predict_proba(inp1)[0]

            # print(abs(pre0 - pre1)[0]

            # if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
            #         and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
            if (abs(out0 - out1) > threshold and (tuple(np.delete(inp0[0], sensitive_param - 1)) not in global_disc_inputs)
                    and (tuple(np.delete(inp0[0], sensitive_param - 1)) not in local_disc_inputs)):
                local_disc_inputs.add(tuple(tuple(np.delete(inp0[0],sensitive_param - 1))))
                local_disc_inputs_list.append(inp0.tolist()[0])
                # print(pre0, pre1)
                # print(out1, out0)

                # print("Percentage discriminatory inputs - " + str(
                #     float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
                # print("Total Inputs are " + str(len(tot_inputs)))
                # print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))

                return 2* abs(out1 - out0) + 1
                # return abs(pre0-pre1)
    # return abs(pre0-pre1)
    return 2* abs(out1 - out0) + 1

    # return not abs(out0 - out1) > threshold
    # for binary classification, we have found that the
    # following optimization function gives better results


def xai_fair_testing(max_global, max_local):

    global log

    config = data_config[dataset]
    feature_names = config.feature_name
    class_names = config.class_name
    sens_name = config.sens_name[sensitive_param]
    params = config.params
    input_bounds = config.input_bounds

    # prepare the testing data and model
    X, Y, input_shape, nb_classes = data[dataset]()

    start = time.time()

    # model_name = classifier_name.split("/")[-1].split("_")[0]
    # file_name = "aequitas_"+dataset+sensitive_param+"_"+model+""
    file_name = f"./baseline/expga/expga_{model_name}_{dataset}{sensitive_param}.txt"
    time_file_name = f'./result/RQ1/expga/expga_{model_name}_{dataset}{sensitive_param}.npy'
    f =open(file_name,"a")
    f.write("iter:"+str(iter)+"------------------------------------------"+"\n"+"\n")
    f.close()

    global_discovery = Global_Discovery()

    train_samples = global_discovery(max_global,params,input_bounds,sensitive_param)
    train_samples = np.array(train_samples)
    # train_samples = X[np.random.choice(X.shape[0], max_global, replace=False)]

    np.random.shuffle(train_samples)


    # print(train_samples.shape)


    explainer = ConstructExplainer(X, feature_names, class_names)

    seed = Searchseed(model, feature_names, sens_name, explainer,train_samples,params,X)

    print('Finish Searchseed')
    for inp in seed:
        inp0 = [int(i) for i in inp]
        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))
        global_disc_inputs.add(tuple(map(tuple, inp0)))
        global_disc_inputs_list.append(inp0.tolist()[0])

    print("Finished Global Search")
    print('length of total input is:' + str(len(tot_inputs)))
    print('length of global discovery is:' + str(len(global_disc_inputs)))
    print('length of global discovery list is:' + str(len(global_disc_inputs_list)))
    with open(file_name, mode='w') as f:
        f.write('length of global discovery is:' + str(len(global_disc_inputs)) + '\n')
        f.write('length of global discovery list is:' + str(len(global_disc_inputs_list)) + '\n')

    np.save(f'./result/expga/global_samples_{dataset}{sensitive_param}_{model_name}_expga.npy',
    np.array(global_disc_inputs_list))

    end = time.time()
    global_time = end - start

    print('Total time:' + str(end - start))


    print("")
    print("Starting Local Search")

    nums = global_disc_inputs_list

    DNA_SIZE = len(input_bounds)
    cross_rate = 0.9
    mutation = 0.05
    iteration = 1000000
    ga = GA(nums=nums, bound=input_bounds, func=evaluate_local, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate, mutation=mutation)  # for random

    times = []
    count = 300
    IDS_count = 1000
    for i in range(iteration):
        ga.evolution()
        end = time.time()
        use_time = end-start
        if use_time >= count:
        # if len(local_disc_inputs_list) >= IDS_count:
            # times.append(end - start)
            # f = open(file_name,"a")
            # f.write("Percentage discriminatory inputs - " + str(
            #     float(len(global_disc_inputs_list) + len(local_disc_inputs_list))
            #     / float(len(tot_inputs)) * 100)+"\n")
            # f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list))+"\n")
            # f.write("Total Inputs are " + str(len(tot_inputs))+"\n")
            # f.write('use time:' + str(end - start)+"\n"+"\n")
            # f.close()

            print("Percentage discriminatory inputs - " + str(float( len(local_disc_inputs_list))
                                                              / float(len(tot_inputs)) * 100))
            print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))

            print('use time:' + str(end - start))
            count +=300
            # IDS_count += 1000

        if total_local_count >= max_local:
            print('local samples 100000')
            break
        if end - start >= 3600:
            break
    np.save(time_file_name, times)
    np.save(f'./result/expga/{dataset}{sensitive_param}_{model_name}_expga.npy',
    np.array(local_disc_inputs_list))

    with open(file_name, mode='w') as f:
        if total_local_count >= max_local:
            f.write('-----------local search 100000-------------\n')
        else:
            f.write('--------Time Out(1h)-----------\n')
        f.write('---------------------------------\n')
        f.write("Total Inputs are " + str(len(tot_inputs)) + '\n')
        f.write("Total Inputs with duplicates are " + str(total_local_count) + '\n')
        f.write("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)) + '\n')
        f.write("Percentage discriminatory inputs - " + str(
        float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100) + '\n')
        f.write(f'use time:{end - start}\n')
        log.append([len(local_disc_inputs_list), end - start, len(local_disc_inputs_list)/(end-start), len(tot_inputs),global_time, end-start-global_time])
    print('---------------------------------------------------')
    print("Total Inputs are " + str(len(tot_inputs)))
    print("Number of discriminatory inputs are " + str(len(local_disc_inputs_list)))
    print("Percentage discriminatory inputs - " + str(
        float(len(local_disc_inputs_list)) / float(len(tot_inputs)) * 100))
    print(f'use time:{end - start}')


def main(argv=None):
    global dataset
    global sensitive_param
    global total_local_count
    global model
    global threshold
    global threshold_l
    global config

    global global_disc_inputs
    global global_disc_inputs_list
    global local_disc_inputs
    global local_disc_inputs_list
    global tot_inputs
    global location

    global log
    global log_index
    
    for dataset_name in dataset_names:
        dataset = dataset_name
        if dataset == 'bank':
            threshold_l = 10
        elif dataset == 'census':
            threshold_l = 7
        else:
            threshold_l = 14  # replace census-7,credit-14,bank-10,meps-14
        threshold = 0
        model = joblib.load(f'./model_info/{model_name}/{dataset}_{model_name}_model.pkl')
        config = data_config[dataset]
        for sens_param in config.sensitive_param:
            # model_path = f'model_info/unawareness/{dataset_name}_{sens_param}_fakemodel.pkl'
            # model = joblib.load(model_path)
            global_disc_inputs = set()
            global_disc_inputs_list = []
            local_disc_inputs = set()
            local_disc_inputs_list = []
            tot_inputs = set()
            location = np.zeros(40)

            sensitive_param = sens_param + 1
            total_local_count = 0
        
            print(f'---------{dataset}-{sensitive_param}-{model_name}-------------:')
            xai_fair_testing(max_global=1000,
                            max_local=100000)
            log_index.append(f'{dataset_name} {sensitive_param}')
    log = [[np.round(x, 3) for x in row] for row in log]
    df = pd.DataFrame(log, columns=column_index, index=log_index)
    df.to_csv(log_path, index=True)

if __name__ == '__main__':
    log_path = f'baseline/expga/{model_name}_logging_data.csv'
    log = []
    log_index = []
    column_index = ['NID', "Time", "NPS", "Total Number", "global time", "local time"]
    main()