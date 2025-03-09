import numpy as np
from tqdm import tqdm
import random
import torch
from model.cluster import seed_test_input, cluster
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
import time
from deap import base, creator, tools, algorithms
from deap.algorithms import varOr
from scipy.spatial.distance import cdist
import random
import os
import matplotlib.pyplot as plt
import pickle


# 创建一个最大化多目标优化问题
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

class NSGA2:
    def __init__(self, data, tree, select_strategy, bounds, sens_param_index, model, TC, local_disc_inputs, local_disc_inputs_list, total_samples, inverse_func, mu=50, lambda_=100, cxpb=0.3, mutpb=0.7, ngen=50):
        self.data = data
        self.bounds = bounds  # 边界条件
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.sens_param_index = sens_param_index
        self.model = model
        # self.original_node = original_node
        self.tree = tree
        self.kmeans = tree
        self.select_strategy = select_strategy
        self.local_disc_inputs = local_disc_inputs
        self.local_disc_inputs_list = local_disc_inputs_list
        self.total_samples = total_samples
        self.TC = TC
        self.inverse_func = inverse_func
        self.count = int(self.TC.IDS_number / 1000) * 1000 + 1000
        # self.avg_distances = []
        self.latent_vector_list = []
        # self.temp_time = 0
        # self.temp_count = 0

        # 初始化DEAP工具箱
        self.toolbox = base.Toolbox()

        # 注册个体生成函数到工具箱
        self.toolbox.register("individual", self.create_individual, data=self.data)

        # 注册评估函数到工具箱
        # self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("batch_evaluate", self.batch_evaluate)

        # 定义交叉和变异操作
        self.toolbox.register("mate", self.custom_crossover)
        self.toolbox.register("mutate", self.mutate, indpb=self.mutpb)

        # 定义选择操作
        self.toolbox.register("select", tools.selNSGA2)

        self.toolbox.register("custom_algorithm", self.custom_eaMuPlusLambda)



    # 定义个体生成函数，从数据集中随机选择一个数据点作为个体，并应用边界条件
    def create_individual(self, data):
        individual = creator.Individual(random.choice(data))
        # # 应用边界条件
        # for i in range(len(individual)):
        #     individual[i] = max(min(individual[i], self.bounds[i][1]), self.bounds[i][0])
        return individual
    
    def custom_crossover(self, ind1, ind2):
        # 随机选择交叉点
        crossover_point = random.randint(1, len(ind1) - 1)
        
        # 执行单点交叉
        for i in range(crossover_point, len(ind1)):
            temp = ind1[i]
            ind1[i] = ind2[i]
            ind2[i] = temp

        return ind1, ind2

    # 定义变异操作，同样要应用边界条件
    def mutate(self, individual, indpb):
        for i in range(len(individual)):
            # if i < 128:

            if random.random() < indpb:
            # individual[i] += random.randint(-5, 5)  # 可以根据需要调整变异的范围
            # # individual[i] += random.uniform(-1, 1)
            # individual[i] = round(individual[i])
            # individual[i] = max(min(individual[i], self.bounds[i][1]), self.bounds[i][0])
                individual[i] = np.random.normal(loc=0, scale=1)
        
            # else:
            #     individual[i] = 0
        # if random.random() < indpb:
        #     index = random.randint(128, len(individual) - 1)
        #     individual[index] = 1
        return [individual]

    # 定义评估函数，根据个体的数据点进行评估
    def evaluate(self, individual):
        # 返回一个元组，包含多个目标函数值
        input_bounds = self.bounds
        sensitive_param = self.sens_param_index
        max_diff = 0  # 累积最大差异度
        is_ids = False
        is_repeated = False
        latent = [i for i in individual]
        latent = np.array(latent).astype(np.float32)

        # start = time.time()
        inp0 = self.inverse_func(latent).astype(int)
        # end = time.time()

        # torch.cuda.synchronize()
        # self.temp_time += end - start
        # self.temp_count += 1
        
        # out0 = np.argmax(self.model.predict(inp0))
        out0 = self.model.predict(inp0)
        self.total_samples.add(tuple(np.delete(inp0[0], sensitive_param)))
        # outs = []
        # outs.append(out0)
        inps = [inp0[0]]

        # sen_range = input_bounds[sensitive_param][1] - input_bounds[sensitive_param][0] + 1
        for val in range(input_bounds[sensitive_param][0], input_bounds[sensitive_param][1] + 1):
            if val != inp0[0][sensitive_param]:
                
                inp1 = inp0[0].copy()
                inp1[sensitive_param] = val
                
                inp1 = np.asarray(inp1)
                inps.append(inp1)

                # inp1 = np.reshape(inp1, (1, -1))
                
                # # out1 = np.argmax(self.model.predict(inp1))
                # out1 = self.model.predict(inp1)
                # outs.append(out1)
                # if sen_range < 4:
                #     pre0 = self.model.predict_proba(inp0)[0]
                #     pre1 = self.model.predict_proba(inp1)[0]
                #     curr_diff = max(2 * abs(pre0 - pre1) + abs(pre0 + pre1 - 1))
                #     max_diff = max(max_diff, curr_diff)  # 更新最大差异度
                # if is_ids:
                #     continue
                # else:
                #     if abs(out0 - out1) > 0:
                #         # self.TC.total_IDS_number += 1
                #         is_ids = True
                #         if (tuple(np.delete(inp0[0], sensitive_param)) not in self.local_disc_inputs):
                #             self.local_disc_inputs.add(tuple(np.delete(inp0[0], sensitive_param)))
                #             self.local_disc_inputs_list.append(inp0.tolist()[0])
                #             self.latent_vector_list.append(latent)
                #             self.TC.IDS_number += 1
                #             if(self.TC.IDS_number >= self.count):
                #                 self.TC.times.append(time.time() - self.TC.start_time)
                #                 self.count += 1000
                #             # if not (self.TC.IDS_number == len(self.local_disc_inputs_list)):
                #             #     pass
                #         else:
                #             is_repeated = True
        outs = self.model.predict(inps)
        outs_set = set([pred for pred in outs])
        if len(outs_set) > 1:
            is_ids= True
            if (tuple(np.delete(inp0[0], sensitive_param)) not in self.local_disc_inputs):
                self.local_disc_inputs.add(tuple(np.delete(inp0[0], sensitive_param)))
                self.local_disc_inputs_list.append(inp0.tolist()[0])
                self.TC.IDS_number += 1
                # if(self.TC.IDS_number >= self.count):
                #     self.TC.times.append(time.time() - self.TC.start_time)
                #     self.count += 1000
            else:
                is_repeated = True


        if is_repeated:
            distance = -1
            distance_metric = -1
        else:
            if self.select_strategy == 'kmeans':
                distance_metric = np.round(np.min(self.kmeans.transform(latent.reshape(1,-1))), 3)
            else:
                distances, indices = self.tree.query(latent.reshape(1,-1), k=4)
                distance_metric = np.round(np.average(distances[0][1:]), 3)
                # avg_distance = np.average(distances[0][1:])
            # self.avg_distances.append(avg_distance)
        outs = np.array(outs)
        unique, counts = np.unique(outs, return_counts=True)
        probabilities = counts / len(outs)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        # if sen_range < 4:
        #     return entropy, distance
        # else:
        return entropy, distance_metric
        # if is_ids:
        #     return 3, 0
        # else:
        #     return 1, 0
    
    def batch_evaluate(self, individuals):
        fitness = []
        latent_vectors = [[i for i in inp] for inp in individuals]
        latent_vectors = np.array(latent_vectors).astype(np.float32)
        inps = self.inverse_func(latent_vectors)
        for inp, lv in zip(inps, latent_vectors):
            # 返回一个元组，包含多个目标函数值
            input_bounds = self.bounds
            sensitive_param = self.sens_param_index
            max_diff = 0  # 累积最大差异度
            is_ids = False
            is_repeated = False

            # start = time.time()
            inp0 = np.reshape(inp.astype(int), (1, -1))
            # end = time.time()

            # torch.cuda.synchronize()
            # self.temp_time += end - start
            # self.temp_count += 1
            
            # out0 = np.argmax(self.model.predict(inp0))
            # out0 = self.model.predict(inp0)
            self.total_samples.add(tuple(np.delete(inp0[0], sensitive_param)))
            # outs = []
            # outs.append(out0)
            
            inps = [inp0[0]]
            # sen_range = input_bounds[sensitive_param][1] - input_bounds[sensitive_param][0] + 1
            for val in range(input_bounds[sensitive_param][0], input_bounds[sensitive_param][1] + 1):
                if val != inp0[0][sensitive_param]:
                    
                    inp1 = inp0[0].copy()
                    inp1[sensitive_param] = val
                    inps.append(inp1)
                    # inp1 = np.reshape(inp1, (1, -1))
                    
                    # # out1 = np.argmax(self.model.predict(inp1))
                    # out1 = self.model.predict(inp1)
                    # outs.append(out1)
                    # # if sen_range < 4:
                    # #     pre0 = self.model.predict_proba(inp0)[0]
                    # #     pre1 = self.model.predict_proba(inp1)[0]
                    # #     curr_diff = max(2 * abs(pre0 - pre1) + abs(pre0 + pre1 - 1))
                    # #     max_diff = max(max_diff, curr_diff)  # 更新最大差异度
                    # if is_ids:
                    #     continue
                    # else:
                    #     if abs(out0 - out1) > 0:
                    #         # self.TC.total_IDS_number += 1
                    #         is_ids = True
                    #         if (tuple(np.delete(inp0[0], sensitive_param)) not in self.local_disc_inputs):
                    #             self.local_disc_inputs.add(tuple(np.delete(inp0[0], sensitive_param)))
                    #             self.local_disc_inputs_list.append(inp0.tolist()[0])
                    #             self.latent_vector_list.append(lv)
                    #             self.TC.IDS_number += 1
                    #             if(self.TC.IDS_number >= self.count):
                    #                 self.TC.times.append(time.time() - self.TC.start_time)
                    #                 self.count += 1000
                    #             # if not (self.TC.IDS_number == len(self.local_disc_inputs_list)):
                    #             #     pass
                    #         else:
                    #             is_repeated = True
            
            outs = self.model.predict(inps)
            outs_set = set([pred for pred in outs])
            if len(outs_set) > 1:
                is_ids = True
                if (tuple(np.delete(inp0[0], sensitive_param)) not in self.local_disc_inputs):
                    self.local_disc_inputs.add(tuple(np.delete(inp0[0], sensitive_param)))
                    self.local_disc_inputs_list.append(inp0.tolist()[0])
                    self.latent_vector_list.append(lv)
                    self.TC.IDS_number += 1
                    # if(self.TC.IDS_number >= self.count):
                    #     self.TC.times.append(time.time() - self.TC.start_time)
                    #     self.count += 1000
                else:
                    is_repeated = True
                                
            if is_ids:
                distance = -1
                distance_metric = -1
            else:
                if self.select_strategy == 'kmeans':
                    distance_metric = np.round(np.min(self.kmeans.transform(lv.reshape(1,-1))), 3)
                else:
                    distances, indices = self.tree.query(lv.reshape(1,-1), k=4)
                    distance_metric = np.round(np.average(distances[0][1:]), 3)
                    # avg_distance = np.average(distances[0][1:])
                # self.avg_distances.append(avg_distance)
            # outs = np.array(outs)
            unique, counts = np.unique(outs, return_counts=True)
            probabilities = counts / len(outs)
            entropy = -np.sum(probabilities * np.log2(probabilities))
            # if sen_range < 4:
            #     return entropy, distance
            # else:
            fitness.append((entropy, distance_metric))
            # if is_ids:
            #     return 3, 0
            # else:
            #     return 1, 0
        return fitness

    def custom_eaMuPlusLambda(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        if invalid_ind is not None:
            fitnesses = toolbox.batch_evaluate(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Vary the population
            offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.batch_evaluate(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)

            # Select the next generation population
            population[:] = toolbox.select(population + offspring, mu)

            # Update the statistics with the new population
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

        return population, logbook

    # 运行NSGA-II算法
    def run(self):
        # 从数据集中随机选择个体来构建初始种群
        initial_population = [self.toolbox.individual() for _ in range(self.mu)]

        # 计算初始种群的适应度值
        # fitnesses = map(self.toolbox.evaluate, initial_population)
        # fitnesses = self.toolbox.batch_evaluate(initial_population)
        # for ind, fit in zip(initial_population, fitnesses):
        #     ind.fitness.values = fit

        # 运行NSGA-II算法
        # algorithms.eaMuPlusLambda(initial_population, self.toolbox, self.mu, self.lambda_, self.cxpb, self.mutpb, self.ngen, stats=None, halloffame=None, verbose=False)
        self.toolbox.custom_algorithm(initial_population, self.toolbox, self.mu, self.lambda_, self.cxpb, self.mutpb, self.ngen, stats=None, halloffame=None, verbose=False)

        # 返回非支配解集
        pareto_front = tools.sortNondominated(initial_population, len(initial_population), first_front_only=True)[0]
        print('total_generate_num', len(self.total_samples))
        print('IDS_num',len(self.local_disc_inputs), self.TC.IDS_number)
        print('Percentage discriminatory inputs:',len(self.local_disc_inputs)/len(self.total_samples))

        # counts, bin_edges, _ = plt.hist(np.array(self.avg_distances), bins=10, alpha=0.5, visible = False)
        # print('count:',counts)
        # print('edges:',bin_edges)
        # print(f'逆函数平均耗时:{self.temp_time/self.temp_count} s')
        return self.local_disc_inputs, self.local_disc_inputs_list, self.latent_vector_list, self.total_samples, self.TC

class BoundaryFair(object):
    def __init__(self, data_config, sens_param_index, input_type, TimingAndCount, projection_model, select_strategy):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_model = projection_model
        self.data_config = data_config
        self.sens_param_index = sens_param_index
        self.input_type = input_type
        if input_type == 'numpy':
            self.vector_list = self._generate_vector_numpy_list(max_vector_num = 30, max_iter=200, threshold=0.5)
        else:
            self.vector_list = self._generate_vector_tensor_list(max_vector_num = 30, max_iter=200, threshold=0.5)
        self.total_num_inquire = 0
        self.total_generate_num = 0
        self.total_samples = set()
        self.TC = TimingAndCount
        self.dataset_name = ''
        self.select_strategy = select_strategy
        
    def inverse_func(self, latent_vector):
        #转为二维torch张量
        latent_vector = torch.tensor(latent_vector, device=self.projection_model._device)
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)

        if len(latent_vector) == 1:
            latent_vector = latent_vector.repeat(2,1)
            sample = self.projection_model.generate_samples_from_latent_vector(latent_vector)[0].reshape(1, -1)
        else:
            sample = self.projection_model.generate_samples_from_latent_vector(latent_vector)
        return sample #二维np数组


    def _generation_random_samples(self, model, num_gen):
        samples, latent_vector = self.projection_model.sample_and_latent(num_gen)
        # gen_samples = self._clip(samples)
        gen_samples = samples
        predictions = model.predict(gen_samples)
        return gen_samples, predictions, latent_vector.detach().cpu().numpy()
    
    def _ray_boundary_search_numpy(self, model, start_sample, vector, max_iter=100,distence_theta=0.005):
        iter_num = 0
        inf_num = 63355
        flag_is_boundary_point = False
        sens_param_index = self.sens_param_index
        #确定最大的系数
        # sample_speace_boudary = self._clip(np.round(start_sample +  inf_num * vector).reshape(1,-1))
        # max_sigma = np.delete(((sample_speace_boudary - start_sample) / vector), sens_param_index).max()
        max_sigma = 1000

        start = np.copy(start_sample)
        end = np.copy(start_sample)
        for sigma in range(1, int(np.ceil(max_sigma + 1))):
            end = start + sigma * vector
            end = self._clip(end)
            test = np.round(end) #取整
            if not (self._is_disc_input(test, model)):
                max_sigma = sigma
                flag_is_boundary_point = True
                break
        if not (flag_is_boundary_point):
            return end, flag_is_boundary_point
        else:
            end = start + max_sigma * vector
            start = end - vector
            while(iter_num < max_iter and np.linalg.norm(start - end) > distence_theta):
                middle = (start + end)/2
                middle = self._clip(middle)
                test = np.round(middle) #取整
                if not (self._is_disc_input(test, model)):
                    end = middle
                    flag_is_boundary_point = True
                else:
                    start = middle
                iter_num += 1
        boundary_sample = self._clip(data = np.round(start).squeeze())       
        return boundary_sample, flag_is_boundary_point

    def distance_measure(self, ids_inputs, model, data_config):
        # ids_inputs without sensitive param
        len_norm = []
        for input_bound in data_config.input_bounds:
            len_norm.append(input_bound[1] - input_bound[0])
        len_norm = np.array(len_norm)
        sens_param_index = self.sens_param_index
        dim = self.data_config.params
        
        sample_size = min(10000, len(ids_inputs))
        min_dis_list = []

        random_sample = ids_inputs[np.random.choice(ids_inputs.shape[0], sample_size, replace=False)]
        random_sample = np.insert(random_sample, sens_param_index, data_config.input_bounds[sens_param_index][0], axis = 1)
        vector_list = self._generate_random_n_vector(dim, dim, sens_param_index)
        for ids in tqdm(random_sample):
            distance_list = []
            for vector in vector_list:
                boundary_sample1, flag_is_boundary_point1 = self._ray_boundary_search_numpy(model, ids, vector)
                boundary_sample2, flag_is_boundary_point2 = self._ray_boundary_search_numpy(model, ids, -vector)
                if flag_is_boundary_point1 and flag_is_boundary_point2:
                    vector = boundary_sample2 - boundary_sample1
                    norm_vector = vector / len_norm
                    distance_list.append(np.linalg.norm(norm_vector))

            min_dis = np.min(distance_list)
            min_dis_list.append(min_dis)
        random_sample = np.delete(random_sample, sens_param_index, axis=1)
        return random_sample, np.array(min_dis_list)
    
    def _find_disc_samples(self, model, sens_param_index, all_samples, data_dict, dataset_name, random_or_cluster = 0):
        self.dataset_name = dataset_name
        start_time = time.time()
        self.TC.start_time = start_time
        # #------------------Random--------------------
        # random_seeds = set()
        # random_seeds_list = []
        # random_latent_vector = set()

        # all_samples, Y, latent_vector = self._generation_random_samples(model, 100000)
        # for sample, lv in zip(all_samples, latent_vector):
        #     if self._is_disc_input(sample, model):
        #         if tuple(np.delete(sample, sens_param_index)) not in random_seeds:
        #             random_seeds.add(tuple(np.delete(sample, sens_param_index)))
        #             random_seeds_list.append(sample)
        #             random_latent_vector.add(tuple(lv))
        #             self.TC.IDS_number += 1
        # end_time = time.time()
        # disc_inputs = random_seeds_list
        # self.Local_time = end_time - start_time
        # self.total_generate_num = 100000
        # return disc_inputs, self.total_generate_num, self.TC
    
        global_seeds, global_seed_list, global_latent_vector = self._global_search(model, random_or_cluster, dataset_name, sens_param_index, all_samples, data_dict)
        end_time = time.time()
        print("Finished Global Search")
        print('length of global discovery is:' + str(len(global_seeds)))
        print('Total time:' + str(end_time - start_time))
        self.TC.Global_time = end_time - start_time
        
        local_disc_inputs, local_disc_inputs_list = self._local_generation(np.array(list(global_latent_vector)), model, sens_param_index)
        # disc_inputs = [np.array(ids) for ids in list(local_disc_inputs)]
        disc_inputs = local_disc_inputs_list
        end_time = time.time()
        print("Finished local Search")
        print('length of local discovery is:' + str(len(local_disc_inputs)))
        print('Success Rate:', self.TC.IDS_number/self.total_generate_num)
        print('Total time:' + str(end_time - start_time))
        self.TC.Local_time = end_time - start_time - self.TC.Global_time
        # if not os.path.exists(f'result/global/'):
        #     os.makedirs('result/global/')
        # # np.save(f'result/global/{dataset_name}{sens_param_index}_{round(self.TC.IDS_number/self.total_generate_num,2)}.npy',np.array(global_seed_list))
        return disc_inputs, self.total_generate_num, self.TC
    
    def _global_search(self, model, random_or_cluster, dataset_name, sens_param_index, all_samples, data_dict):
        init_samples_num = 50 #初始全局搜索样本点
        max_end_samples = 20 #全局搜索中每个初始样本的引导样本数
        # max_iter_global_search = 20 #全局搜索中对每个初始样本进行多少次搜索
        # max_iter_local_search = 5 #局部搜索中的搜索
        max_iter_binary_search = 1000 #二分搜索的最大迭代次数
        distance_theta = 0.005
        cluster_num = 4

        global_seeds = set()
        global_seeds_list = []
        global_latent_vector = set()
        
        if random_or_cluster == 0:
            all_samples, Y, latent_vector = self._generation_random_samples(model, 1000)
            
            data_dict = {0:[],1:[]}
            if Y.ndim > 1:
                for i in range(len(all_samples)):
                    if Y[i][0] == 1:
                        data_dict[0].append(i)
                    else:
                        data_dict[1].append(i)
            else:
                for i in range(len(all_samples)):
                    if Y[i] == 0:
                        data_dict[0].append(i)
                    else:
                        data_dict[1].append(i)
            
            inputs_index = [random.randint(0, len(all_samples)-1) for _ in range(init_samples_num)]
        else:
            #进行聚类
            clf = cluster(dataset_name=dataset_name, X = all_samples, cluster_num=cluster_num, model_path='./model_info/clusters/')
            clusters = [np.where(clf.labels_==i) for i in range(cluster_num)]

            #进行全局搜索
            inputs_index = seed_test_input(clusters, init_samples_num)
        max_end_samples = min(max_end_samples, len(data_dict[0]))
        max_end_samples = min(max_end_samples, len(data_dict[1]))
        
        # #------------------Global random--------------------
        # for sample, lv in zip(all_samples, latent_vector):
        #     # if self._is_disc_input(sample, model):
        #     if tuple(np.delete(sample, sens_param_index)) not in global_seeds:
        #         global_seeds.add(tuple(np.delete(sample, sens_param_index)))
        #         global_seeds_list.append(sample)
        #         global_latent_vector.add(tuple(lv))
        # return global_seeds, global_seeds_list, global_latent_vector
    
        for sample, lv in zip(all_samples, latent_vector):
            if self._is_disc_input(sample, model):
                if tuple(np.delete(sample, sens_param_index)) not in global_seeds:
                    global_seeds.add(tuple(np.delete(sample, sens_param_index)))
                    global_seeds_list.append(sample)
                    global_latent_vector.add(tuple(lv))
        return global_seeds, global_seeds_list, global_latent_vector

    
    def _local_generation(self, global_seeds_array:np.ndarray, model, sens_param_index):
        local_disc_inputs_list = []
        local_disc_inputs = set()

        # -----------------------------local random----------------------------------
        indpd = 0.05
        if len(global_seeds_array) > 100:
            init_sampls = global_seeds_array[np.random.choice(global_seeds_array.shape[0], 100, replace=False)]
        else:
            init_sampls =  global_seeds_array[np.random.choice(global_seeds_array.shape[0], 100, replace=True)]
        for sample in tqdm(init_sampls):
            lv_list = []
            for _ in range(1000):
                lv = np.copy(sample)
                for j in range(len(lv)):
                    if random.random() < indpd:
                        lv[j] = np.random.normal(loc=0, scale=1)
                lv_list.append(lv)
            lv_list = np.array(lv_list).astype(np.float32)
            inps = self.inverse_func(lv_list).astype(int)
            for inp in inps:
                self.total_generate_num += 1
                if self._is_disc_input(inp, model) and (tuple(np.delete(inp, sens_param_index)) not in local_disc_inputs):
                    local_disc_inputs.add(tuple(np.delete(inp, sens_param_index)))
                    local_disc_inputs_list.append(inp.tolist())
                    self.TC.IDS_number += 1
        return local_disc_inputs, local_disc_inputs_list

        #以全局seed为基础,进行聚类
        kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto')
        
        tree_nodes = global_seeds_array.copy()
        population_size = 500
        total_num = 100000
        ngen = 100
        cxpb = 0.9
        mutpb = 0.1
        select_strategy = self.select_strategy
        
        for i in tqdm(range(int(total_num/(2*population_size)/ngen))):
            if select_strategy == 'kmeans':
                kmeans.fit(tree_nodes)
            if len(tree_nodes) < population_size:
                # tree_nodes = tree_nodes.tolist()
                # now_nodes = random.sample(tree_nodes, population_size - len(tree_nodes))
                # tree_nodes.extend(now_nodes)
                # tree_nodes = np.array(tree_nodes)
                tree_nodes = tree_nodes[np.random.choice(tree_nodes.shape[0], population_size, replace=True)]
            tree = BallTree(tree_nodes, metric= 'minkowski')
            distances, indices = tree.query(tree_nodes, k=6)
            avg_distances = [np.average(np.array(distance[1:])) for distance in distances]
            sorted_indices = np.argsort(avg_distances)[::-1]
            index = sorted_indices[range(population_size)]
            # candidatas = random.sample(tree_nodes.tolist(), population_size)
            # kmeans_distance_list = []
            # for tree_node in tree_nodes:
            #     tree_node = np.reshape(tree_node, (1, -1))
            #     kmeans_distance_list.append(np.min(kmeans.transform(tree_node)))
            # sorted_indices = np.argsort(kmeans_distance_list)[::-1]
            # index = sorted_indices[range(population_size)]
            candidatas = []
            candidatas.extend(tree_nodes[index])
            if select_strategy == 'kmeans':
                ga = NSGA2(candidatas, kmeans, select_strategy, self.data_config.input_bounds, self.sens_param_index, model, self.TC, local_disc_inputs, local_disc_inputs_list, self.total_samples, self.inverse_func,
                       cxpb= cxpb, mutpb=mutpb, mu = population_size, lambda_=2 * population_size, ngen=ngen)
            else:
                ga = NSGA2(candidatas, tree, select_strategy, self.data_config.input_bounds, self.sens_param_index, model, self.TC, local_disc_inputs, local_disc_inputs_list, self.total_samples, self.inverse_func,
                       cxpb= cxpb, mutpb=mutpb, mu = population_size, lambda_=2 * population_size, ngen=ngen)
            local_disc_inputs, local_disc_inputs_list, latent_vector, self.total_samples, self.TC = ga.run()
            self.total_generate_num = len(self.total_samples)
            global_seeds_array = np.concatenate((global_seeds_array, np.array(latent_vector)),axis=0)
            percent = 1
            num_to_select = min(int(len(global_seeds_array) * percent), 5000)
            tree_nodes = global_seeds_array[np.random.choice(global_seeds_array.shape[0], num_to_select, replace=False)]
        tree = BallTree(tree_nodes, metric= 'minkowski')
        distances, indices = tree.query(tree_nodes, k=6)
        avg_distances = [np.average(np.array(distance[1:])) for distance in distances]
        print('After:',np.max(avg_distances))
        return local_disc_inputs, local_disc_inputs_list

    def fitness(self, individual, model, original_node):
        input_bounds = self.data_config.input_bounds
        sensitive_param = self.sens_param_index
        max_diff = 0  # 累积最大差异度
        for val in range(input_bounds[sensitive_param][0], input_bounds[sensitive_param][1] + 1):
            if val != individual[sensitive_param]:
                inp1 = [i for i in individual]
                inp1[sensitive_param] = val

                inp0 = np.asarray(individual)
                inp0 = np.reshape(inp0, (1, -1))

                inp1 = np.asarray(inp1)
                inp1 = np.reshape(inp1, (1, -1))

                out0 = model.predict(inp0)
                out1 = model.predict(inp1)

                pre0 = model.predict_proba(inp0)[0]
                pre1 = model.predict_proba(inp1)[0]
                curr_diff = max(pre0 - pre1)
                max_diff = max(max_diff, curr_diff)  # 更新最大差异度

        distance = np.linalg.norm(individual - original_node)

        return max_diff, distance 

    def _generate_random_n_vector(self, vector_num, dimensions, sens_param_index):
        random.seed(int(time.time()))
        dimensions = dimensions - 1
        #Fibonacci Sphere,10维以下
        vectors = []
        if dimensions < 10:
            phi = np.pi * (3. - np.sqrt(5.))  # 黄金角度

            for i in range(vector_num):
                y = 1 - (i / float(vector_num - 1)) * 2  # 在[-1, 1]之间均匀分布
                radius = np.sqrt(1 - y * y)
                theta = phi * i

                point = np.zeros(dimensions)
                point[0] = np.cos(theta) * radius
                point[1] = y
                point[2] = np.sin(theta) * radius

                if dimensions > 3:
                    remaining_dims = dimensions - 3
                    random_angles = np.random.uniform(0, 2 * np.pi, remaining_dims)
                    for j in range(remaining_dims):
                        point[j + 3] = np.cos(random_angles[j]) * radius
                        radius = np.sqrt(1 - point[j + 3] * point[j + 3])

                point_with_zero = np.insert(point, sens_param_index, 0)
                vectors.append(point_with_zero)
        #Spherical 4-Polytope，适用于更高纬度
        else:
            for _ in range(vector_num):
                point = np.random.normal(size=dimensions)
                point /= np.linalg.norm(point)
                point_with_zero = np.insert(point, sens_param_index, 0)
                vectors.append(point_with_zero)
        
        return vectors
    
    def _is_disc_input_by_latent(self, latent_vector, model):
        sample = self.inverse_func(latent_vector)
        sens_param_index = self.sens_param_index
        sens_param_bounds = self.data_config.input_bounds[sens_param_index]
        tags = set()
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        elif sample.ndim > 2:
            raise('sample wrong!')
        for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
            sample[0][sens_param_index] = sens_value
            # tags.add(np.argmax(model.predict(sample)))
            tags.add(tuple(model.predict(sample)))
            self.total_num_inquire += 1
        if len(tags) > 1:
            return True
        else:
            return False

    def _is_disc_input(self, input_sample, model):
        sens_param_index = self.sens_param_index
        sens_param_bounds = self.data_config.input_bounds[sens_param_index]
        tags = set()
        sample = input_sample.copy()
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        elif sample.ndim > 2:
            raise('sample wrong!')
        for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
            sample[0][sens_param_index] = sens_value
            # tags.add(np.argmax(model.predict(sample)))
            tags.add(tuple(model.predict(sample)))
            self.total_num_inquire += 1
        if len(tags) > 1:
            return True
        else:
            return False

    def _binary_search_latent_numpy(self, model, start_latent, end_latent, max_iter=1000, distence_theta=0.005):
        iter_num = 0
        inf_num = 63355
        flag_is_adv = False
        start = np.copy(start_latent)
        end = np.copy(end_latent)
        
        start_sample = self.inverse_func(start)
        end_sample = self.inverse_func(end)
        
        # start_tag = np.argmax(model.predict(start))
        # end_tag = np.argmax(model.predict(end))
        start_tag = model.predict(start_sample)
        end_tag = model.predict(end_sample)
        self.total_num_inquire += 2

        if(start_tag == end_tag):
            direction_vector = (end - start) / np.linalg.norm(end - start)
            sigma = inf_num
            new_end = start + sigma * direction_vector
            new_end_sample = self.inverse_func(new_end)
            new_end_sample = self._clip(new_end_sample)
            # new_end_tag = np.argmax(model.predict(new_end))
            new_end_tag = model.predict(new_end_sample)
            self.total_num_inquire += 1

            if new_end_tag == start_tag:
                return start, flag_is_adv
            else:
                while(iter_num < max_iter and np.linalg.norm(start - end, ord=2) > distence_theta):
                    sigma /= 2
                    middle = start + sigma * direction_vector
                    test = self.inverse_func(middle)
                    test = self._clip(test)
                    test = np.round(test) #取整
                    self.total_num_inquire += 1
                    # if not (np.argmax(model.predict(test)) == start_tag):
                    if not (model.predict(test) == start_tag):
                        end = middle
                        flag_is_adv = True
                    else:
                        start = middle
                    iter_num += 1
        else:
            while(iter_num < max_iter and np.linalg.norm(start - end, ord=2) > distence_theta):
                middle = (start + end)/2.0
                test = self.inverse_func(middle)
                test = np.round(test) #取整
                self.total_num_inquire += 1
                # if not (np.argmax(model.predict(test)) == start_tag):
                if not (model.predict(test) == start_tag):
                    end = middle
                    flag_is_adv = True
                else:
                    start = middle
                iter_num += 1
        
        boundary_latent = end
        return boundary_latent, flag_is_adv

    def _binary_search_numpy(self, model, start_sample, end_sample, max_iter=1000, distence_theta=0.005):
        iter_num = 0
        inf_num = 63355
        flag_is_adv = False
        start = np.copy(start_sample)
        end = np.copy(end_sample)
            
        # start_tag = np.argmax(model.predict(start))
        # end_tag = np.argmax(model.predict(end))
        start_tag = model.predict(start)
        end_tag = model.predict(end)
        self.total_num_inquire += 2

        if(start_tag == end_tag):
            direction_vector = (end - start) / np.linalg.norm(end - start)
            sigma = inf_num
            new_end = start + sigma * direction_vector
            new_end = self._clip(new_end)
            # new_end_tag = np.argmax(model.predict(new_end))
            new_end_tag = model.predict(new_end)
            self.total_num_inquire += 1

            if new_end_tag == start_tag:
                return start, flag_is_adv
            else:
                while(iter_num < max_iter and np.linalg.norm(start - end, ord=2) > distence_theta):
                    sigma /= 2
                    middle = start + sigma * direction_vector
                    middle = self._clip(middle)
                    test = np.round(middle) #取整
                    self.total_num_inquire += 1
                    # if not (np.argmax(model.predict(test)) == start_tag):
                    if not (model.predict(test) == start_tag):
                        end = middle
                        flag_is_adv = True
                    else:
                        start = middle
                    iter_num += 1
        else:
            while(iter_num < max_iter and np.linalg.norm(start - end, ord=2) > distence_theta):
                middle = (start + end)/2.0
                test = np.round(middle) #取整
                self.total_num_inquire += 1
                # if not (np.argmax(model.predict(test)) == start_tag):
                if not (model.predict(test) == start_tag):
                    end = middle
                    flag_is_adv = True
                else:
                    start = middle
                iter_num += 1
        
        boundary_sample = self._clip(data = np.round(end).squeeze())
        return boundary_sample, flag_is_adv

    def _binary_search(self, model, start_sample:torch.tensor, end_sample:torch.tensor, max_iter=1000, distence_theta=0.005):
        '''
        
        '''
        iter_num = 0
        inf_num = 63355
        flag_is_adv = False
        start = torch.clone(start_sample)
        end = torch.clone(end_sample)
        start_tag = model.predict(start)
        end_tag = model.predict(end)
        self.total_num_inquire += 2

        if(start_tag == end_tag):
            direction_vector = (end - start) / torch.norm(end - start)
            sigma = inf_num
            end = start + sigma * direction_vector
            end = self._clip(end)
            end_tag = model.predict(end)
            self.total_num_inquire += 1

            if end_tag == start_tag:
                return start, flag_is_adv
            else:
                while(iter_num < max_iter and torch.dist(start, end) > distence_theta):
                    sigma /= 2
                    middle = start + sigma * direction_vector
                    middle = self._clip(middle)
                    test = torch.round(middle) #取整
                    self.total_num_inquire += 1
                    if not (model.predict(test) == start_tag):
                        end = middle
                        flag_is_adv = True
                    else:
                        start = middle
                    iter_num += 1
        else:
            while(iter_num < max_iter and torch.dist(start, end) > distence_theta):
                middle = (start + end)/2.0
                test = torch.round(middle) #取整
                self.total_num_inquire += 1
                if not (model.predict(test) == start_tag):
                    end = middle
                    flag_is_adv = True
                else:
                    start = middle
                iter_num += 1
        
        boundary_sample = self._clip(data = torch.round(end).squeeze())
        return boundary_sample, flag_is_adv

    def _clip(self, data):
        data_config = self.data_config
        if data.ndim > 1:
            for d in data:
                for i in range(data_config.params):
                    d[i] = data_config.input_bounds[i][1] if d[i] > data_config.input_bounds[i][1] else d[i]
                    d[i] = data_config.input_bounds[i][0] if d[i] < data_config.input_bounds[i][0] else d[i]       
        else:
            for i in range(data_config.params):
                data[i] = data_config.input_bounds[i][1] if data[i] > data_config.input_bounds[i][1] else data[i]
                data[i] = data_config.input_bounds[i][0] if data[i] < data_config.input_bounds[i][0] else data[i]       
        return data
    
    def _generate_vector_numpy_list(self, max_vector_num, max_iter = 100, threshold = 0.5):
        vector_count = 0
        vector_list = []
        sens_param_index = self.sens_param_index

        for i in range(max_iter):
            #随机扰动以生成足够多的方向
            random_vector = np.random.randn(self.data_config.params)
            random_vector[sens_param_index] = 0
            normalized_vector = random_vector / np.linalg.norm(random_vector)

            # 对于空列表，直接添加新向量
            if len(vector_list) == 0:
                vector_list.append(normalized_vector)
                vector_count += 1
            else:
                # 计算新张量与当前张量列表的余弦相似度
                # 首先计算当前向量和向量列表中的每个向量的内积
                dot_products = np.dot(vector_list, normalized_vector)

                # 计算当前向量的范数
                norm_current_vector = np.linalg.norm(normalized_vector)

                # 计算向量列表中每个向量的范数
                norm_vector_list = np.linalg.norm(vector_list, axis=1)

                # 计算余弦相似度
                cosine_similarities = dot_products / (norm_vector_list * norm_current_vector)

                # 检查余弦相似度是否小于阈值
                if (cosine_similarities <= threshold).all():
                    vector_list.append(normalized_vector)
                    vector_count += 1
            if vector_count >= max_vector_num:
                break
        return vector_list

    def _generate_vector_tensor_list(self, max_vector_num, max_iter, threshold):
        vector_count = 0
        vector_list = []
        sens_param_index = self.sens_param_index
        print('search for more random vectors')
        for i in tqdm(range(max_iter)):
            #随机扰动以生成足够多的方向
            random_vector = np.random.randn(self.data_config.params)
            random_vector[sens_param_index] = 0
            normalized_vector = random_vector / np.linalg.norm(random_vector)
            normalized_vector = torch.from_numpy(normalized_vector).to(self.device, dtype=torch.float32)
            # 对于空列表，直接添加新张量
            if len(vector_list) == 0:
                vector_list.append(normalized_vector)
                vector_count += 1
            else:
                # 计算新张量与当前张量列表的余弦相似度
                similarities = torch.nn.functional.cosine_similarity(normalized_vector, torch.stack(vector_list), dim=1)

                # 检查余弦相似度是否小于阈值
                if (similarities <= threshold).all():
                    vector_list.append(normalized_vector)
                    vector_count += 1
            if vector_count >= max_vector_num:
                break
        return vector_list

    def __call__(
            self,
            model,
            sens_param_index,
            all_samples,
            data_dict,
            dataset_name,
            random_or_cluster,
    ):
        return self._find_disc_samples(
                                    model=model,
                                    sens_param_index=sens_param_index,
                                    all_samples=all_samples,
                                    data_dict=data_dict,
                                    dataset_name=dataset_name,
                                    random_or_cluster=random_or_cluster,
                                    )

if __name__ == '__main__':
    test = BoundaryFair()
    a = np.array([1,4,5,78,9]).astype(np.float32)
    sens_param_index = 8 #census,index from 0, 8 for gender, 9 for race
    