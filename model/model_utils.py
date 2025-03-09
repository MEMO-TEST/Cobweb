# from model.MLP import MLPClassifier
from data_preprocess.bank import bank_data
from data_preprocess.credit import credit_data
from data_preprocess.census import census_data
from data_preprocess.mydataset import create_data_loaders
from sklearn.model_selection import train_test_split
from data_preprocess.config import bank,credit,census

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
# from model.ensemble_training import EnsembleClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

from model.ensemble_training import EnsembleClassifier

import os
from utils.logging_handler import logger_model_train, logger_model_retrain
import joblib


def dataPreprocess(X, Y, Opt_sampler=None):
    if Y.ndim > 1:
        nb_classes = Y.shape[1]
        temp = []
        for y in Y:
            if int(y[0]) == 1:
                temp.append(0)
            else:
                temp.append(1)
        Y = np.array(temp, dtype=float)
    else:
        nb_classes = np.max(Y) + 1
    #数据过采样
    if Opt_sampler == 'oversampler':
        oversampler = RandomOverSampler()
        X, Y = oversampler.fit_resample(X, Y)
        logger_model_train.info('Oversample!')
    elif Opt_sampler == 'undersampler':
        undersampler = RandomUnderSampler()
        X, Y = undersampler.fit_resample(X, Y)
        logger_model_train.info('Undersample!')
    # 进行数据集的分割
    train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)  

    # # 计算训练集中每个类别的样本数量和比例
    # train_class_counts = np.sum(train_labels, axis=0)
    # train_class_proportions = train_class_counts / len(train_labels)

    # # 计算测试集中每个类别的样本数量和比例
    # test_class_counts = np.sum(test_labels, axis=0)
    # test_class_proportions = test_class_counts / len(test_labels)

    # # 打印训练集和测试集中每个类别的样本数量和比例
    # for i, class_label in enumerate(range(nb_classes)):
    #     print("Class:", class_label)
    #     print("Train Count:", train_class_counts[i])
    #     print("Train Proportion:", train_class_proportions[i])
    #     print("Test Count:", test_class_counts[i])
    #     print("Test Proportion:", test_class_proportions[i])
    #     print("---------------------")
    return train_data, train_labels, test_data, test_labels

def trainModel(dataset_name, model_name, X, Y, save_path, optsampler=None):
    train_data, train_labels, test_data, test_labels = dataPreprocess(X, Y, optsampler)
    if model_name == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=[64,32,16,8,4],max_iter=1000)
    elif model_name == 'SVM':
        clf = SVC(probability=True)
    elif model_name == 'NB':
        clf = GaussianNB()
    elif model_name == 'RF':
        clf = RandomForestClassifier()
    elif model_name == 'KNN':
        clf = KNeighborsClassifier()
    clf.fit(train_data, train_labels)
    joblib.dump(clf, save_path)
    pred_labels = clf.predict(test_data)

    # test_labels = np.argmax(test_labels, axis=1)
    # pred_labels = np.argmax(pred_labels, axis=1)
    report = classification_report(test_labels, pred_labels)
    print(report)
    # score = clf.score(test_data, test_labels)
    # print('score:', score)
    logging_data = [f'{dataset_name}_{model_name}_train:', report]
    logger_model_train.info("\n".join(logging_data))


def trainEnsembleModel(dataset_name, sens_param_index,save_path):
    ensembleclassifier = EnsembleClassifier()
    ensembleclassifier.train(dataset_name, sens_param_index,save_path)
def retrainModelWithSelection(dataset_name, model_name, approach_name,ensemble_model_path, ids, min_dis, save_path, sens_param_index, data_config, dataset,scaler = None, optsampler = None):
    # ids without sensitive_param
    # select 5% of individual discriminatory instances generated for data augmentation
    # then retrain the original models

    ensemble_clf = joblib.load(ensemble_model_path)
    sens_param_bounds = data_config.input_bounds[sens_param_index]
    
    if model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=[64,32,16,8,4],max_iter=1000)
    elif model_name == 'SVM':
        model = SVC(probability=True)
    elif model_name == 'NB':
        model = GaussianNB()
    elif model_name == 'RF':
        model = RandomForestClassifier()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()

    X, Y, input_shape, nb_classes = dataset()
    # y = [0 if temp[0] == 1 else 1 for temp in Y]
    
    ids_aug = []
    num_aug = int(min(int(len(ids) * 0.1), len(X)))
    sorted_indices = np.argsort(min_dis)
    sampled_indices = sorted_indices[:num_aug]
    for i in sampled_indices:
        for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
            ids_aug.append(np.insert(ids[i],sens_param_index,sens_value))
    # ids_aug = [ids[i] for i in sampled_indices]
    ids_aug = np.array(ids_aug)
    
    # sens_param_index = data_config[dataset_name].sensitive_param

    labels_vote = ensemble_clf.predict(np.delete(ids_aug, sens_param_index, axis=1))
    labels_vote = [[0,1] if label_vote==1 else [1,0] for label_vote in labels_vote]
    X = np.append(X, ids_aug, axis=0)
    Y = np.append(Y, labels_vote, axis=0)
    if scaler != None:
        X = scaler.transform(X)
    X_train, y_train, x_test, y_test = dataPreprocess(X, Y, Opt_sampler=optsampler)
    model.fit(X_train, y_train)
    pred_labels = model.predict(x_test)
    y_test = np.argmax(y_test, axis=1)
    pred_labels = np.argmax(pred_labels, axis=1)
    report = classification_report(y_test, pred_labels)
    print(report)
    joblib.dump(model, save_path)
    logging_data = [f'{dataset_name}_{sens_param_index}_{model_name}_{approach_name}_retrain:', report]
    logger_model_retrain.info("\n".join(logging_data))

def retrainModel(dataset_name, model_name, approach_name,ensemble_model_path, ids, save_path, sens_param_index, data_config, dataset, scaler = None, optsampler = None):
    #  Must remove the sensitive params of ids firstly!!!
    #  randomly sample 5% of individual discriminatory instances generated for data augmentation
    #  retrain the original models
    
    ensemble_clf = joblib.load(ensemble_model_path)
    sens_param_bounds = data_config.input_bounds[sens_param_index]
    
    if model_name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=[64,32,16,8,4],max_iter=1000)
    elif model_name == 'SVM':
        model = SVC(probability=True)
    elif model_name == 'NB':
        model = GaussianNB()
    elif model_name == 'RF':
        model = RandomForestClassifier()
    elif model_name == 'KNN':
        model = KNeighborsClassifier()

    X, Y, input_shape, nb_classes = dataset()
    Y = [0 if temp[0] == 1 else 1 for temp in Y]
    original_X = X
    original_Y = Y
    ids_aug = []
    num_aug = int(min(int(len(X) * 0.1), len(ids)))
    random_ids = ids[np.random.choice(ids.shape[0], num_aug, replace=False)]
    # 衡量歧视程度，然后挑选最佳的

    for selected_ids in random_ids:
        for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
            ids_aug.append(np.insert(selected_ids,sens_param_index,sens_value))
    ids_aug = np.array(ids_aug)
    
    # sens_param_index = data_config[dataset_name].sensitive_param

    labels_vote = ensemble_clf.predict(np.delete(ids_aug, sens_param_index, axis=1))
    # labels_vote = [[0,1] if label_vote==1 else [1,0] for label_vote in labels_vote]
    X = np.append(X, ids_aug, axis=0)
    Y = np.append(Y, labels_vote, axis=0)
    if scaler != None:
        X = scaler.transform(X)
    # X_train, y_train, x_test, y_test = dataPreprocess(X, Y, Opt_sampler=optsampler)
    # model.fit(X_train, y_train)
    # pred_labels = model.predict(x_test)
    # y_test = np.argmax(y_test, axis=1)
    # pred_labels = np.argmax(pred_labels, axis=1)
    # report = classification_report(y_test, pred_labels)
    
    model.fit(X, Y)
    pred_labels = model.predict(original_X)
    # original_Y = np.argmax(original_Y, axis=1)
    # pred_labels = np.argmax(pred_labels, axis=1)
    report = classification_report(original_Y, pred_labels)
    print(report)
    joblib.dump(model, save_path)
    logging_data = [f'{dataset_name}_{sens_param_index}_{model_name}_{approach_name}_retrain:', report]
    logger_model_retrain.info("\n".join(logging_data))