import numpy as np
from tqdm import tqdm
import random
import torch
from model.cluster import seed_test_input, cluster
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
import time
from deap import base, creator, tools, algorithms
# from deap.algorithms import varOr
from scipy.spatial.distance import cdist
import random
import os
import matplotlib.pyplot as plt

import cProfile
import math
import pickle
import cloudpickle

profiler = cProfile.Profile()

# Create a maximization multi-objective optimization problem
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

def varOr(population, toolbox, lambda_, cxpb, mutpb):
    r"""Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population."""

    offspring = []
    for _ in range(int(lambda_/2)):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            # ind1, ind2 = random.sample(population, 2)
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
            offspring.append(ind1)
            offspring.append(ind2)
        else:                           # Apply reproduction
            offspring.extend(random.sample(population, 2))
    
    temp_inds = []
    for temp_ind in offspring:
        ind, = toolbox.mutate(temp_ind)
        del ind.fitness.values
        temp_inds.append(ind)

    # return offspring
    return temp_inds

class NSGA2:
    def __init__(self, data, tree, select_strategy, bounds, sens_param_index, model, TC, local_disc_inputs, local_disc_inputs_list, total_samples, inverse_func, mu=50, lambda_=100, cxpb=0.3, mutpb=0.7, ngen=50):
        self.data = data
        self.bounds = bounds
        self.mu = mu
        self.lambda_ = lambda_
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.ngen = ngen
        self.sens_param_index = sens_param_index
        self.model = model
        self.tree = tree
        self.kmeans = tree
        self.select_strategy = select_strategy
        self.local_disc_inputs = local_disc_inputs
        self.local_disc_inputs_list = local_disc_inputs_list
        self.total_samples = total_samples
        self.TC = TC
        self.inverse_func = inverse_func
        self.count = int(self.TC.IDS_number / 1000) * 1000 + 1000
        self.latent_vector_list = []


        # init
        self.toolbox = base.Toolbox()

        self.toolbox.register("individual", self.create_individual, data=self.data)

        self.toolbox.register("batch_evaluate", self.batch_evaluate)

        self.toolbox.register("mate", self.custom_crossover)
        self.toolbox.register("mutate", self.mutate, indpb=self.mutpb)

        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("custom_algorithm", self.custom_eaMuPlusLambda)



    def create_individual(self, data):
        individual = creator.Individual(random.choice(data))
        return individual
    
    # crossover
    def custom_crossover(self, ind1, ind2):
        crossover_point = random.randint(1, len(ind1) - 1)
        for i in range(crossover_point, len(ind1)):
            temp = ind1[i]
            ind1[i] = ind2[i]
            ind2[i] = temp

        return ind1, ind2

    # mutation
    def mutate(self, individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = np.random.normal(loc=0, scale=1)
        return [individual]

    
    def batch_evaluate(self, individuals):
        fitness = []
        latent_vectors = [[i for i in inp] for inp in individuals]
        latent_vectors = np.array(latent_vectors).astype(np.float32)
        inps = self.inverse_func(latent_vectors)
        for inp, lv in zip(inps, latent_vectors):
            input_bounds = self.bounds
            sensitive_param = self.sens_param_index
            is_ids = False
            inp0 = np.reshape(inp.astype(int), (1, -1))

            self.total_samples.add(tuple(np.delete(inp0[0], sensitive_param)))

            
            inps = [inp0[0]]
            for val in range(input_bounds[sensitive_param][0], input_bounds[sensitive_param][1] + 1):
                if val != inp0[0][sensitive_param]:
                    
                    inp1 = inp0[0].copy()
                    inp1[sensitive_param] = val
                    inps.append(inp1)
            
            outs = self.model.predict(inps)
            outs_set = set([pred for pred in outs])
            if len(outs_set) > 1:
                is_ids = True
                if (tuple(np.delete(inp0[0], sensitive_param)) not in self.local_disc_inputs):
                    self.local_disc_inputs.add(tuple(np.delete(inp0[0], sensitive_param)))
                    self.local_disc_inputs_list.append(inp0.tolist()[0])
                    self.latent_vector_list.append(lv)
                    self.TC.IDS_number += 1
                                
            if is_ids:
                distance_metric = -1
            else:
                if self.select_strategy == 'kmeans':
                    distance_metric = np.round(np.min(self.kmeans.transform(lv.reshape(1,-1))), 3)
                else:
                    distances, indices = self.tree.query(lv.reshape(1,-1), k=4)
                    distance_metric = np.round(np.average(distances[0][1:]), 3)

            unique, counts = np.unique(outs, return_counts=True)
            probabilities = counts / len(outs)
            entropy = -np.sum(probabilities * np.log2(probabilities))

            fitness.append((entropy, distance_metric))

        return fitness

    def custom_eaMuPlusLambda(self, population, toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=True):
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
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

    def run(self):
        initial_population = [self.toolbox.individual() for _ in range(self.mu)]
        self.toolbox.custom_algorithm(initial_population, self.toolbox, self.mu, self.lambda_, self.cxpb, self.mutpb, self.ngen, stats=None, halloffame=None, verbose=False)

        pareto_front = tools.sortNondominated(initial_population, len(initial_population), first_front_only=True)[0]
        print('total_generate_num', len(self.total_samples))
        print('IDS_num',len(self.local_disc_inputs), self.TC.IDS_number)
        print('Percentage discriminatory inputs:',len(self.local_disc_inputs)/len(self.total_samples))

        return self.local_disc_inputs, self.local_disc_inputs_list, self.latent_vector_list, self.total_samples, self.TC


class FairnessTest(object):
    def __init__(self, data_config, sens_param_index, TimingAndCount, projection_model, select_strategy):
        self.projection_model = projection_model
        self.data_config = data_config
        self.sens_param_index = sens_param_index
        self.total_num_inquire = 0
        self.total_generate_num = 0
        self.total_samples = set()
        self.TC = TimingAndCount
        self.dataset_name = ''
        self.select_strategy = select_strategy
        
    def inverse_func(self, latent_vector):
        latent_vector = torch.tensor(latent_vector, device=self.projection_model._device)
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)

        if len(latent_vector) == 1:
            latent_vector = latent_vector.repeat(2,1)
            sample = self.projection_model.generate_samples_from_latent_vector(latent_vector)[0].reshape(1, -1)
        else:
            sample = self.projection_model.generate_samples_from_latent_vector(latent_vector)
        return sample #2d nparray


    def _generation_random_samples(self, model, num_gen):
        samples, latent_vector = self.projection_model.sample_and_latent(num_gen)
        gen_samples = samples
        predictions = model.predict(gen_samples)
        return gen_samples, predictions, latent_vector.detach().cpu().numpy()
    

    def _find_disc_samples(self, model, sens_param_index, all_samples, data_dict, dataset_name, max_global, max_local):
        self.dataset_name = dataset_name
        start_time = time.time()
        self.TC.start_time = start_time
        global_seeds, global_seed_list, global_latent_vector = self._global_search(model, max_global, sens_param_index, all_samples, data_dict)
        end_time = time.time()
        print("Finished Global Search")
        print('length of global discovery is:' + str(len(global_seeds)))
        print('Total time:' + str(end_time - start_time))
        self.TC.Global_time = end_time - start_time
        
        local_disc_inputs, local_disc_inputs_list = self._local_generation(np.array(list(global_latent_vector)), model, stop_steps=max_global * max_local)
        disc_inputs = local_disc_inputs_list
        end_time = time.time()
        print("Finished local Search")
        print('length of local discovery is:' + str(len(local_disc_inputs)))
        print('Success Rate:', self.TC.IDS_number/self.total_generate_num)
        print('Total time:' + str(end_time - start_time))
        self.TC.Local_time = end_time - start_time - self.TC.Global_time
    
        return disc_inputs, self.total_generate_num, self.TC
    
    def _global_search(self, model, max_global, sens_param_index, all_samples, data_dict):
        global_seeds = set()
        global_seeds_list = []
        global_latent_vector = set()

        
        all_samples, Y, latent_vector = self._generation_random_samples(model, max_global * 10)
    
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

        for sample, lv in zip(all_samples, latent_vector):
            if self._is_disc_input(sample, model):
                if tuple(np.delete(sample, sens_param_index)) not in global_seeds:
                    global_seeds.add(tuple(np.delete(sample, sens_param_index)))
                    global_seeds_list.append(sample)
                    global_latent_vector.add(tuple(lv))
        return global_seeds, global_seeds_list, global_latent_vector

    def _local_generation(self, global_seeds_array:np.ndarray, model, stop_steps):
        start_time = time.time()
        local_disc_inputs_list = []
        local_disc_inputs = set()
        kmeans = KMeans(n_clusters=8, random_state=42, n_init='auto')
        
        tree_nodes = global_seeds_array.copy()
        population_size = 500
        ngen = 100
        cxpb = 0.9
        mutpb = 0.2
        select_strategy = self.select_strategy

        #spatial uniform initialization 
        if len(tree_nodes) < population_size:
            tree_nodes = tree_nodes[np.random.choice(tree_nodes.shape[0], population_size, replace=True)]
        else:
            tree_nodes = tree_nodes[self.max_min_distance_selection(tree_nodes, population_size)]
        
        for i in tqdm(range(int(stop_steps/(2*population_size)/ngen))):
            if select_strategy == 'kmeans':
                kmeans.fit(tree_nodes)
            if len(tree_nodes) < population_size:
                tree_nodes = tree_nodes[np.random.choice(tree_nodes.shape[0], population_size, replace=True)]
            tree = BallTree(tree_nodes, metric= 'minkowski')
            distances, indices = tree.query(tree_nodes, k=6)
            avg_distances = [np.average(np.array(distance[1:])) for distance in distances]
            sorted_indices = np.argsort(avg_distances)[::-1]
            index = sorted_indices[range(population_size)]

            candidates = []
            candidates.extend(tree_nodes[index])
            if select_strategy == 'kmeans':
                ga = NSGA2(candidates, kmeans, select_strategy, self.data_config.input_bounds, self.sens_param_index, model, self.TC, local_disc_inputs, local_disc_inputs_list, self.total_samples, self.inverse_func,
                       cxpb= cxpb, mutpb=mutpb, mu = population_size, lambda_=2 * population_size, ngen=ngen)
            else:
                ga = NSGA2(candidates, tree, select_strategy, self.data_config.input_bounds, self.sens_param_index, model, self.TC, local_disc_inputs, local_disc_inputs_list, self.total_samples, self.inverse_func,
                       cxpb= cxpb, mutpb=mutpb, mu = population_size, lambda_=2 * population_size, ngen=ngen)
            local_disc_inputs, local_disc_inputs_list, latent_vector, self.total_samples, self.TC = ga.run()
            self.total_generate_num = len(self.total_samples)
            global_seeds_array = np.concatenate((global_seeds_array, np.array(latent_vector)),axis=0)
            percent = 1
            num_to_select = min(int(len(global_seeds_array) * percent), 5000)
            tree_nodes = global_seeds_array[np.random.choice(global_seeds_array.shape[0], num_to_select, replace=False)]

        return local_disc_inputs, local_disc_inputs_list

    # the maximum and minimum distance selection strategy
    def max_min_distance_selection(self, candidates, population_size):
        selected_indices = [np.random.randint(len(candidates))]
        remaining_indices = set(range(len(candidates))) - set(selected_indices)

        for _ in range(population_size - 1):
            max_min_dist = -np.inf
            best_index = None
            
            for idx in remaining_indices:
                # Calculate the distance from the current point to the nearest point in the selected point set
                min_dist = min(np.linalg.norm(candidates[idx] - candidates[sel]) for sel in selected_indices)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_index = idx

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return selected_indices


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
            tags.add(tuple(model.predict(sample)))
            self.total_num_inquire += 1
        if len(tags) > 1:
            return True
        else:
            return False
        
    def __call__(
            self,
            model,
            sens_param_index,
            all_samples,
            data_dict,
            dataset_name,
            max_global,
            max_local
    ):
        return self._find_disc_samples(
                                    model=model,
                                    sens_param_index=sens_param_index,
                                    all_samples=all_samples,
                                    data_dict=data_dict,
                                    dataset_name=dataset_name,
                                    max_global = max_global,
                                    max_local = max_local
                                    )
    