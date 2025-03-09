import pandas as pd
import pickle

from limictgan.synthesizers.ctgan import CTGANSynthesizer
# from ctgan import CTGAN
from limictgan.synthesizers.base import random_state

import numpy as np
import torch


from data_preprocess.bank import bank_data
from data_preprocess.credit import credit_data
from data_preprocess.census import census_data
from data_preprocess.meps import meps_data
from data_preprocess.config import bank,credit,census,meps
from tqdm import tqdm
import random
import time
import joblib

time_trans = 0
time_compute = 0
time_inverse = 0
count_samples = 0

dataset_dict = {"census":census_data, "credit":credit_data, "bank":bank_data, 'meps':meps_data}
data_config = {"census":census, "credit":credit, "bank":bank, "meps": meps}

class CTGANModel(CTGANSynthesizer):
    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=0, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True):
        super().__init__(embedding_dim, generator_dim, discriminator_dim, generator_lr, generator_decay, discriminator_lr, discriminator_decay, batch_size, discriminator_steps, log_frequency, verbose, epochs, pac, cuda)
    
    def is_true_sample(self, samples, condition_column=None, condition_value=None):
        
        samples = samples.loc[np.repeat(samples.index, 10)].reset_index(drop=True)

        samples = self._transformer.transform(samples)

        # 如果提供了条件列和条件值，则生成条件向量
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, len(samples)
            )
        else:
            condition_vec, col, opt = self._data_sampler.generate_cond_from_data(samples)
        
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples.astype("float32")).to(self._device)
        elif not isinstance(samples, torch.Tensor):
            raise TypeError("`samples` must be a numpy array or torch tensor.")
        
        condition_vec = torch.from_numpy(condition_vec.astype("float32")).to(self._device)

        # 拼接条件向量到输入样本
        samples = torch.cat([samples, condition_vec], dim=1)

        # # 当前行数
        # num_rows = samples.size(0)

        # # 计算需要移除的行数，使总行数是 10 的倍数
        # remove_rows = num_rows % 10

        # # 如果需要移除行数不为 0，则移除这些行
        # if remove_rows != 0:
        #     samples = samples[:-remove_rows]

        self._discriminator.eval()

        # 通过辨别器计算置信度
        with torch.no_grad():  # 禁用梯度计算
            raw_scores = self._discriminator(samples)  # 辨别器输出 logits
            confidence = torch.sigmoid(raw_scores)  # 使用 Sigmoid 转化为 [0, 1] 区间的概率
            
        return [True if c > 0.5 else False for c in confidence]

    # @random_state
    def sample_and_latent(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        self._generator.eval()
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        latent_vector = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake, phase='generate')
            latent_vector.append(fakez)
            data.append(fakeact.detach().cpu().numpy())

        latent_vector = torch.cat(latent_vector, dim=0)
        latent_vector = latent_vector[:n]
        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data).to_numpy(), latent_vector

    @random_state
    def sample_from_latent_vector(self, latent_vector:torch.tensor):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        # global time_compute
        # global time_trans
        # global count_samples
        # global time_inverse
        
        self._generator.eval()
        n = len(latent_vector)
        
        data = []
        count = 0
        with torch.no_grad():
            while count < n:
                fakez = latent_vector[count : count + self._batch_size if count + self._batch_size < n else n]
                count += self._batch_size

                # start = time.time()

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake, phase="generate")

                # torch.cuda.synchronize()
                # end = time.time()
                # time_compute += (end - start)
                
                data.append(fakeact.detach().cpu().numpy())
                # new_end = time.time()
                # time_trans += (new_end - end)

                # count_samples += 1

        data = np.concatenate(data, axis=0)
        # data = data[:n]
        # start = time.time()
        data = self._transformer.inverse_transform(data).to_numpy()
        # end = time.time()
        # time_inverse += (end - start)
        return data
    
    # def generate_samples(self, num_samples):
    #     """
    #     从随机噪声生成样本
    #     :param num_samples: 生成样本的数量
    #     :return: 生成的样本 latent_vector:torch.tensor
    #     """
    #     samples_list = []
    #     latent_vector_list = []
    #     for _ in tqdm(range(num_samples)):
    #         samples, latent_vector = self.sample_and_latent(2)
    #         samples_list.append(samples[0])
    #         latent_vector_list.append(latent_vector[0])
    #     return np.array(samples_list), torch.stack(latent_vector_list)
    
    def generate_samples_from_latent_vector(self, latent_vector: torch.tensor):
        """
        从随机噪声生成样本
        :param latent_vector: 随机噪声
        :return: 生成的样本
        """
        if len(latent_vector) == 1:
            latent_vector = latent_vector.repeat(2,1)
        samples = self.sample_from_latent_vector(latent_vector)
        return samples


if __name__ == "__main__":
    # dataset_names = ['bank']
    dataset_names = ['bank','census', 'credit', 'meps']
    
    for dataset_name in dataset_names:
        count_samples = 0
        time_compute = 0
        time_trans = 0
        X, Y, input_shape, nb_classes = dataset_dict[dataset_name]()
        data = pd.DataFrame(X, columns=data_config[dataset_name].feature_name)
        # discrete_columns = data_config[dataset_name].feature_name
        discrete_columns_indeces = data_config[dataset_name].discrete_columns_indeces
        discrete_columns = [data_config[dataset_name].feature_name[i] for i in discrete_columns_indeces]
        # 训练模型
        model = CTGANModel(verbose=True)
        model.fit(data, discrete_columns, epochs=300)

        # 保存模型
        model.save(f"model_info/CTGAN/{dataset_name}_ctgan_model.pkl")

        del model

        # 读取模型
        # model : CTGANModel
        # model = CTGANModel.load(f"model_info/CTGAN/{dataset_name}_ctgan_model.pkl")
        # test_model = joblib.load(f'model_info/MLP/{dataset_name}_MLP_model.pkl')
        # 生成样本
        # samples, latent_vector = model.generate_samples(num_samples=1000)
        # lv = latent_vector.detach().cpu().numpy()
        # new_samples = []
        # start_time = time.time()
        # for l in latent_vector:
        #     new_sample = model.generate_samples_from_latent_vector(l.unsqueeze(0))
        #     new_samples.append(new_sample[0])
        # end_time = time.time()
        # print('Time for single:', np.round(end_time - start_time, 3))
        # print("new sampels:", np.array(new_samples).shape)

        # start_time = time.time()
        # new_samples = model.generate_samples_from_latent_vector(latent_vector)
        # end_time = time.time()
        # print('Time for group:', np.round(end_time - start_time, 3))
        # print('new sampels:', new_samples.shape)

        ### 测试多次生成的误差
        # samples, latent_vector = model.sample_and_latent(20000)
        # np.save(f'test/{dataset_name}.npy', samples.astype(int))
        # flags_list = []
        # original_indices = np.arange(latent_vector.shape[0])
        # for i in range(20):
        #     shuffled_indices = np.random.permutation(original_indices)
        #     shuffled_latent_vector = latent_vector[shuffled_indices]
        #     new_samples = model.generate_samples_from_latent_vector(shuffled_latent_vector)
        #     flags = test_model.predict(new_samples)
        #     flags = flags[np.argsort(shuffled_indices)]
        #     flags_list.append(flags)
        # flags = np.array(flags_list)
        # unique_count = [len(np.unique(flags[:, col])) for col in range(flags.shape[1])]
        # # print('unique_count:', unique_count)
        # print('count:',sum(size > 1 for size in unique_count))

        # new_samples = np.array(new_samples)
        # # flag_s = np.unique(samples, axis=0).shape[0]
        # # flag_lv = np.unique(lv, axis=0).shape[0]
        # # flag_ns = np.unique(new_samples, axis=0).shape[0]
        # # print(flag_s, flag_lv, flag_ns)
        # row_equality = np.all(samples == new_samples, axis=1)
        # inconsistent_count = np.sum(~row_equality)
        # # print(row_equality)
        # print('-------------------------------------')
        # print(inconsistent_count)

        # print(f'-----{dataset_name}--------')
        # print('count_samples', count_samples)
        # print(f'平均计算时间:{np.round(time_compute/count_samples, 3)} s')
        # print(f'平均传输时间:{time_trans/count_samples} s')
        # print(f'平均转换时间:{np.round(time_inverse/count_samples)} s')
        
        # with open("test_time.txt", "a", encoding="utf-8") as file:
        #     file.write(f'-----{dataset_name}--------\n')
        #     file.write(f'平均计算时间:{time_compute/count_samples} s\n')
        #     file.write(f'平均传输时间:{time_trans/count_samples} s\n')
        #     file.write(f'平均转换时间:{time_inverse/count_samples} s\n')


