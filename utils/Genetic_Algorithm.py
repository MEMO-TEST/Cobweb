import numpy as np
import pandas as pd
from tqdm import tqdm
import time

#classifier_name = 'Random_Forest_standard_unfair.pkl'
#model = joblib.load(classifier_name)

class GA():
    # input:
    #     nums: m * n  n is nums_of x, y, z, ...,and m is population's quantity
    #     bound:n * 2  [(min, nax), (min, max), (min, max),...]
    #     DNA_SIZE is binary bit size, None is auto
    def __init__(self, data, bounds, sens_param_index, model, TC, local_disc_inputs, local_disc_inputs_list, total_samples,  cross_rate=0.8, mutation=0.003):
        nums= np.array(data)
        bound = np.array(bounds)
        self.bound = bound
        self.DNA_SIZE = len(bounds)
        self.POP_SIZE = len(nums)
        # POP = np.zeros((*nums.shape, DNA_SIZE))
        # for i in range(nums.shape[0]):
        #     for j in range(nums.shape[1]):
        #         num = int(round((nums[i, j] - bound[j][0]) * ((2 ** DNA_SIZE) / var_len[j])))
        #         POP[i, j] = [int(k) for k in ('{0:0' + str(DNA_SIZE) + 'b}').format(num)]
        # self.POP = POP
        self.POP = nums

        self.copy_POP = nums.copy()
        self.cross_rate = cross_rate
        self.mutation = mutation
        self.func = self.evaluate_local
        # self.importance = imp
        self.model = model
        self.TC = TC
        self.local_disc_inputs = local_disc_inputs
        self.local_disc_inputs_list = local_disc_inputs_list
        self.total_samples = total_samples
        self.sens_param_index = sens_param_index


    # def translateDNA(self):
    #     W_vector = np.array([2 ** i for i in range(self.DNA_SIZE)]).reshape((self.DNA_SIZE, 1))[::-1]
    #     binary_vector = self.POP.dot(W_vector).reshape(self.POP.shape[0:2])
    #     for i in range(binary_vector.shape[0]):
    #         for j in range(binary_vector.shape[1]):
    #             binary_vector[i, j] /= ((2 ** self.DNA_SIZE) / self.var_len[j])
    #             binary_vector[i, j] += self.bound[j][0]
    #     return binary_vector
    def get_fitness(self, non_negative=False):
        # result = self.func(*np.array(list(zip(*self.translateDNA()))))
        result = [self.func(self.POP[i]) for i in range(len(self.POP))]
        if non_negative:
            min_fit = np.min(result, axis=0)
            result -= min_fit
        return result

    def select(self):
        fitness = self.get_fitness()
        fit = [item[0] for item in fitness]
        # print(fit)
        self.POP = self.POP[np.random.choice(np.arange(self.POP.shape[0]), size=self.POP.shape[0], replace=True,
                                             p=fit / np.sum(fit))]
        pop_str = []
        for pop in self.POP:
            temp = []
            for x in pop:
                temp.append(str(x))
            pop_str.append(temp)
        pop_str = ["".join(x) for x in pop_str]
        # print(len(set(pop_str)))

    def crossover(self):
        k=0
        for people in self.POP:
            # imp = [abs(x) for x in self.importance[k]]
            # k += 1
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(0, self.POP.shape[0], size=1)
                cross_points = np.random.randint(0, len(self.bound))
                end_points = np.random.randint(0, len(self.bound)-cross_points)
                people[cross_points:end_points] = self.POP[i_, cross_points:end_points]

            # if np.random.rand() < self.cross_rate:
            #     i_ = np.random.randint(0, self.POP.shape[0], size=1)
            #     n = np.random.randint(0, len(people), size=1)
            #     # n=1
            #     cross_points = np.random.choice(np.arange(len(self.bound)), size=n, replace=False,
            #                                  p=imp / np.sum(imp))
            #     for j in cross_points:
            #         people[j] = self.POP[i_, j]

    def mutate(self):
        for people in self.POP:
            for point in range(self.DNA_SIZE):
                if np.random.rand() < self.mutation:
                    # var[point] = 1 if var[point] == 0 else 1
                    people[point] = np.random.randint(self.bound[point][0],self.bound[point][1])

    def evolution(self):
        self.select()
        self.crossover()
        self.mutate()

    def reset(self):
        self.POP = self.copy_POP.copy()

    def log(self):
        # return pd.DataFrame(np.hstack((self.POP, self.get_fitness())),
        #                     columns=['x{i}' for i in range(len(self.bound))] + ['F'])
        pop_str = []
        for pop in self.POP:
            temp = []
            for x in pop:
                temp.append(str(x))
            pop_str.append(temp)
        return pop_str, self.get_fitness()
    def evaluate_local(self, inp):
        inp0 = [int(i) for i in inp]
        inp0 = np.array(inp0)

        inp0 = np.reshape(inp0, (1, -1))
        out0 = self.model.predict(inp0)

        self.total_samples.add(tuple(np.delete(inp0[0], self.sens_param_index)))
        for val in range(self.bound[self.sens_param_index][0], self.bound[self.sens_param_index][1]+1):
            if val != inp[self.sens_param_index]:
                inp1 = [int(i) for i in inp]
                inp1[self.sens_param_index] = val

                inp1 = np.array(inp1)
                inp1 = np.reshape(inp1, (1, -1))

                out1 = self.model.predict(inp1)

                # pre0 = model.predict_proba(inp0)[0]
                # pre1 = model.predict_proba(inp1)[0]

                # print(abs(pre0 - pre1)[0]

                # if (abs(out0 - out1) > threshold and (tuple(map(tuple, inp0)) not in global_disc_inputs)
                #         and (tuple(map(tuple, inp0)) not in local_disc_inputs)):
                if (abs(out0 - out1) > 0 and (tuple(np.delete(inp0[0], self.sens_param_index)) not in self.local_disc_inputs)):
                    self.local_disc_inputs.add(tuple(tuple(np.delete(inp0[0], self.sens_param_index))))
                    self.local_disc_inputs_list.append(inp0.tolist()[0])
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
    def run(self, max_iter=1000):
        IDS_count = 1000
        for _ in tqdm(range(max_iter)):
            self.evolution()
            end = time.time()
            if len(self.local_disc_inputs_list) >= IDS_count:
                self.TC.times.append(end - self.TC.start_time)
                IDS_count += 1000
        self.TC.IDS_number = len(self.local_disc_inputs_list)
        self.TC.total_generate_num = len(self.total_samples)
        return self.local_disc_inputs, self.local_disc_inputs_list, self.total_samples, self.TC




# if __name__ == '__main__':
#     nums = [[3,0,10,3,1,6,3,0,1,0,0,40,0],[4,3,20,13,2,5,3,0,0,0,0,50,0],[3,0,14,1,0,4,2,4,1,0,0,80,0],[5,0,5,3,1,0,5,0,0,0,0,40,0]]
#     bound = config.input_bounds
#     # func = lambda x, y: x*np.cos(2*np.pi*y)+y*np.sin(2*np.pi*x)
#     DNA_SIZE = len(bound)
#     cross_rate = 0.7
#     mutation = 0.01
#     ga = GA(nums=nums, bound=bound, func=evaluate_local, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate, mutation=mutation)
#     res = ga.log()
#     print(res)
#     for i in range(10):
#         ga.evolution()
#         res = ga.log()
#         print(res)