import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from data_preprocess.config import bank, census, credit, meps

def pca_plot(data_path, save_path, dimensions = 2, color = 'blue'):
    data = np.load(data_path)
    # 创建PCA对象，指定降维后的维度，如果不指定，默认保留所有主成分
    pca = PCA(n_components=dimensions)
    pca_result = pca.fit_transform(data)
    
    fig = plt.figure(figsize=(8, 6))
    if dimensions == 2:
        ax = fig.add_subplot(111)

        ax.scatter(pca_result[:, 0], pca_result[:, 1], label='Data 1', c='yellow')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA - 2D Visualization')
    elif dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], label='Data 1', c=color)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA - 3D Visualization')
    plt.title('PCA Result')
    plt.savefig(save_path)

if __name__ == '__main__':
    ID = 'BF10'
    if not os.path.exists(f'./result/RQ2PCA/{ID}'):
        os.makedirs(f'./result/RQ2PCA/{ID}')
    method_names = ['BF', 'expga', 'aequitas', 'SG']
    dataset_names = ['bank', 'census', 'credit', 'meps']
    data_config = {"census":census, "credit":credit, "bank":bank, "meps": meps}
    for dataset_name in dataset_names:
        # # dataset_name = 'meps'
        # if dataset_name == 'credit':
        #     sens_param_index = 8
        # elif dataset_name == 'meps':
        #     sens_param_index = 2
        # else:
        #     sens_param_index = 0
        for sens_param_index in data_config[dataset_name].sensitive_param:
            # 读取数据
            
            data_BF = np.load(f'./result/BF_GA_random/{dataset_name}{sens_param_index}_MLP_BF_GA_random.npy')
            data_BF = np.delete(data_BF, [sens_param_index], axis = 1)
            data_expga = np.load(f'./result/expga/{dataset_name}{sens_param_index+1}_MLP_expga.npy')
            data_expga = np.delete(data_expga, [sens_param_index], axis = 1)
            data_aequitas = np.load(f'./result/aequitas/{dataset_name}{sens_param_index+1}_MLP_aequitas.npy')
            data_SG = np.load(f'./result/SG/{dataset_name}{sens_param_index+1}_MLP_SG.npy')
            #删除敏感属性列
            total_data = np.concatenate((data_BF, data_expga, data_aequitas, data_SG), axis=0)

            # 创建PCA对象，指定降维后的维度，如果不指定，默认保留所有主成分
            pca = PCA(n_components=2)
            len_data = [len(data_BF), len(data_expga), len(data_aequitas)]
            
            split_index = []
            temp_len = 0
            for l in len_data:
                temp_len += l
                split_index.append(temp_len)

            # 对数据进行主成分分析
            total_pca_result = pca.fit_transform(total_data)
            pca_result_BF, pca_result_expga, pca_result_aequitas, pca_result_SG = np.split(total_pca_result, split_index, axis=0)
            pca_results = [ pca_result_expga, pca_result_BF, pca_result_aequitas, pca_result_SG]
            fig = plt.figure(figsize=(8, 8))
            # for i, (pca_result, method_name) in enumerate(zip(pca_results, method_names)):
            #     ax = fig.add_subplot(2, 2, i+1)
            #     # ax = fig.add_subplot(111, projection='3d')
            #     ax.scatter(pca_result[:, 0], pca_result[:, 1], label= method_name)
            #     ax.set_xlabel('PC1')
            #     ax.set_ylabel('PC2')
            #     ax.set_title(f'PCA - {method_name}')
            colors = ['blue', 'yellow', 'green', 'black']
            ax = fig.add_subplot(111)
            for pca_result, method_name, color in zip(pca_results, method_names, colors):
                ax.scatter(pca_result[:, 0], pca_result[:, 1], label= method_name, c=color)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
            
            
            # ax.scatter(pca_result4[:, 0], pca_result4[:, 1], pca_result4[:, 2], label='Expga', c='yellow')
            # ax.scatter(pca_result1[:, 0], pca_result1[:, 1], pca_result1[:, 2], label='BF', c='blue')
            

            # ax.set_xlabel('PC1')
            # ax.set_ylabel('PC2')
            # ax.set_zlabel('PC3')
            # ax.set_title('PCA - 2D Visualization')
            plt.title('PCA Result')
            # plt.show()
            plt.savefig(f'./result/RQ2PCA/{dataset_name}{sens_param_index}_compare.png')
            # plt.clf()

