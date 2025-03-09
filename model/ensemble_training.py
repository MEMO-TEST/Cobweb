"""
This python file trains the ensemble models for relabeling.
Five classifiers are trained for majority voting, including KNN, MLP, RBF SVM, Random Forest, Naive Bayes.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
from data_preprocess.config import bank,credit,census
from data_preprocess.bank import bank_data
from data_preprocess.credit import credit_data
from data_preprocess.census import census_data
from data_preprocess.meps import meps_data
from sklearn.metrics import classification_report
from utils.logging_handler import logger_model_train
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

class EnsembleClassifier:
    def __init__(self):
        # create classifiers
        self.knn_clf = KNeighborsClassifier()
        self.mlp_clf = MLPClassifier(max_iter=1000)
        self.svm_clf = SVC(probability=True)
        self.rf_clf = RandomForestClassifier()
        self.nb_clf = GaussianNB()


        # ensemble above classifiers for majority voting
        self.eclf = VotingClassifier(estimators=[('knn', self.knn_clf), ('mlp', self.mlp_clf), ('svm', self.svm_clf), ('rf', self.rf_clf), ('nb', self.nb_clf)],
                        voting='soft')
        # self.eclf = EnsembleVoteClassifier(clfs=[knn_clf, knn_clf, svm_clf, rf_clf, nb_clf], weights=[2, 1, 1], voting='soft')


        # set a pipeline to handle the prediction process
        self.clf = Pipeline([('scaler', StandardScaler()),
                        ('ensemble', self.eclf)])
        
        # self.data_config = {"census":census, "credit":credit, "bank":bank}
        self.dataset_dict = {"census":census_data, "credit":credit_data, "bank":bank_data, 'meps':meps_data}
    
    def _dataPreprocess(self, X, Y, split_ratio, Opt_sampler=None):
        nb_classes = Y.shape[1]
        #数据过采样
        if Opt_sampler == 'oversampler':
            oversampler = RandomOverSampler()
            X, Y = oversampler.fit_resample(X, Y)
            logger_model_train.info('Oversample!')
        elif Opt_sampler == 'undersampler':
            undersampler = RandomUnderSampler()
            X, Y = undersampler.fit_resample(X, Y)
            logger_model_train.info('Undersample!')
        if Opt_sampler != None:
            temp = []
            for y in Y:
                if int(y[0]) == 0:
                    temp.append([1, 0])
                else:
                    temp.append([0, 1])
            Y = np.array(temp, dtype=float)
        Y = [0 if label[0]==1 else 1 for label in Y]
        # 进行数据集的分割
        train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=split_ratio, random_state=42, stratify=Y)
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
        # 计算训练集中每个类别的样本数量和比例
        train_class_counts = np.bincount(train_labels)
        train_class_proportions = train_class_counts / len(train_labels)

        # 计算测试集中每个类别的样本数量和比例
        test_class_counts = np.bincount(test_labels)
        test_class_proportions = test_class_counts / len(test_labels)

        # 打印训练集和测试集中每个类别的样本数量和比例
        for class_label in range(len(train_class_counts)):
            print("Class:", class_label)
            print("Train Count:", train_class_counts[class_label])
            print("Train Proportion:", train_class_proportions[class_label])
            print("Test Count:", test_class_counts[class_label])
            print("Test Proportion:", test_class_proportions[class_label])
            print("---------------------")
        return train_data, test_data, train_labels, test_labels

    def train(self, dataset_name, sens_param_index, save_path, opt_sampler=None):
        # train, evaluate and save ensemble models for each dataset
        X, Y, input_shape, nb_classes = self.dataset_dict[dataset_name]()
        model = clone(self.clf)
        # X = np.delete(X, self.data_config[dataset_name].sensitive_param, axis=1)
        X = np.delete(X, sens_param_index, axis=1)
        
        if len(X) > 10000:
            split_ratio = 0.2
        else:
            split_ratio = 0.4
        
        X_train, X_test, y_train, y_test = self._dataPreprocess(X, Y, split_ratio, Opt_sampler=opt_sampler)
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print('score:', score)
        pred_labels = model.predict(X_test)
        report = classification_report(y_test, pred_labels)
        print(report)
        joblib.dump(model, save_path)
        logging_data = [f'{dataset_name}_ensemble_model:', report]
        logger_model_train.info("\n".join(logging_data))

if __name__ == '__main__':
    ensembleclassifier = EnsembleClassifier()
    ensembleclassifier.train(dataset_name='bank', opt_sampler='oversampler')