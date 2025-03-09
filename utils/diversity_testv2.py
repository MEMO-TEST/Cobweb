import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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
    dataset = 'bank'
    if dataset == 'credit':
        sens_param_index = 8
    elif dataset == 'meps':
        sens_param_index = 2
    else:
        sens_param_index = 0
    # 读取数据
    data_BF = np.load(f'./result/{dataset}_BF_GA_random_disc_samples.npy')
    data_expga = np.load(f'./result/expga/{dataset}{sens_param_index+1}_MLP_expga.npy')
    #删除敏感属性列
    data_BF = np.delete(data_BF, sens_param_index, axis=1)
    data_expga = np.delete(data_expga, sens_param_index, axis=1)


    # 创建PCA对象，指定降维后的维度，如果不指定，默认保留所有主成分
    pca = PCA(n_components=1)

    # 对数据进行主成分分析
    pca_result_BF = pca.fit_transform(data_BF)
    pca_result_expga = pca.fit_transform(data_expga)


    # 设置绘图参数
    bins = 20 # 直方图的数量

    sns.set(style='ticks', font_scale=1.7)

    # 绘制直方密度线图
    sns.distplot(pca_result_BF, bins=bins, kde=True, hist=True, label='BF', color='royalblue')
    sns.distplot(pca_result_expga, bins=bins, kde=True, hist=True, label='expga', color='orange', )

    # 设置图例和标签
    # plt.xlabel('X')
    plt.ylabel('Frequency and Density', fontsize=18)

    # plt.title('Comparison of Data Distributions')
    plt.legend()
    plt.title('PCA Result')
    # plt.show()
    plt.savefig(f'./result/picture/{dataset}_compare.png')
    plt.show()

