import numpy as np
import time
import random

# 定义粒子群类
class Particle:
    def __init__(self, position):
        self.position = position
        self.velocity = position
        self.best_position = self.position.copy()
        self.fitness = 0

class PSO:
    def __init__(self, data, tree, bounds, sens_param_index, model, TC, local_disc_inputs, local_disc_inputs_list, total_samples):
        self.data = data
        self.bounds = bounds  # 边界条件
        self.sens_param_index = sens_param_index
        self.model = model
        # self.original_node = original_node
        self.tree = tree
        self.local_disc_inputs = local_disc_inputs
        self.local_disc_inputs_list = local_disc_inputs_list
        self.total_samples = total_samples
        self.TC = TC
        self.count = int(self.TC.IDS_number / 1000) * 1000 + 1000
        # Rastrigin函数的和
    def rastrigin(self, position):
        # 返回一个元组，包含多个目标函数值
        input_bounds = self.bounds
        sensitive_param = self.sens_param_index
        is_ids = False
        inp0 = [int(i) for i in position]
        inp0 = np.asarray(inp0)
        inp0 = np.reshape(inp0, (1, -1))
        # out0 = np.argmax(self.model.predict(inp0))
        out0 = self.model.predict(inp0)
        self.total_samples.add(tuple(np.delete(inp0[0], sensitive_param)))
        outs = []
        outs.append(out0)
        # sen_range = input_bounds[sensitive_param][1] - input_bounds[sensitive_param][0] + 1
        for val in range(input_bounds[sensitive_param][0], input_bounds[sensitive_param][1] + 1):
            if val != position[sensitive_param]:
                
                inp1 = [int(i) for i in position]
                inp1[sensitive_param] = val

                inp1 = np.asarray(inp1)
                inp1 = np.reshape(inp1, (1, -1))
                
                # out1 = np.argmax(self.model.predict(inp1))
                out1 = self.model.predict(inp1)
                outs.append(out1)
                if is_ids:
                    continue
                else:
                    if abs(out0 - out1) > 0:
                        # self.TC.total_IDS_number += 1
                        is_ids = True
                        if (tuple(np.delete(inp0[0], sensitive_param)) not in self.local_disc_inputs):
                            self.local_disc_inputs.add(tuple(np.delete(inp0[0], sensitive_param)))
                            self.local_disc_inputs_list.append(inp0.tolist()[0])
                            self.TC.IDS_number += 1
                            if(self.TC.IDS_number >= self.count):
                                self.TC.times.append(time.time() - self.TC.start_time)
                                self.count += 1000
                            
        if (tuple(np.delete(inp0[0], sensitive_param)) in self.local_disc_inputs):
            distance = -1
        else:
            # distance = np.linalg.norm(position - self.original_node)
            distances, indices = self.tree.query([position], k = 6)
            distance = [np.average(np.array(distance[1:])) for distance in distances][0]

        outs = np.array(outs)
        unique, counts = np.unique(outs, return_counts=True)
        probabilities = counts / len(outs)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return -entropy - 0.01 * distance

    # 粒子群优化算法
    def run(self, num_particles, max_iter):
        # 初始化粒子群
        particles = np.array(random.sample(self.data, num_particles))
        particles = [Particle(p) for p in particles]
        global_best_position = particles[0].best_position.copy()
        global_best_fitness = particles[0].fitness

        for _ in range(max_iter):
            for particle in particles:
                # 更新粒子速度和位置
                inertia = 0.5
                cognitive_coeff = 1.5
                social_coeff = 1.5
                r1, r2 = np.random.rand(), np.random.rand()

                particle.velocity = (inertia * particle.velocity +
                                    cognitive_coeff * r1 * (particle.best_position - particle.position) +
                                    social_coeff * r2 * (global_best_position - particle.position))
                particle.position += particle.velocity

                # 更新粒子的最佳位置和全局最佳位置
                fitness = self.rastrigin(particle.position)
                if fitness < particle.fitness:
                    particle.fitness = fitness
                    particle.best_position = particle.position.copy()

                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()
        print('total samples:', len(self.total_samples))
        print('IDS:', len(self.local_disc_inputs))
        print('Percentage discriminatory inputs:',len(self.local_disc_inputs)/len(self.total_samples))
        return self.local_disc_inputs, self.local_disc_inputs_list, self.total_samples, self.TC
if __name__ == '__main__':
    pass