import pickle
import time
import numpy as np
import sys, os
import copy
import joblib
import logging
import pandas as pd

from data_preprocess.census import census_data
from data_preprocess.bank import bank_data
from data_preprocess.credit import credit_data
from data_preprocess.meps import meps_data
from data_preprocess.config import census, credit, bank, meps
from tqdm import tqdm
import torch
from GAN import CTGANModel

EXP_DIR = "baseline/LIMI/"
dataset_dict = {"census":census_data, "credit":credit_data, "bank":bank_data, 'meps':meps_data}
data_config = {"census":census, "credit":credit, "bank":bank, "meps": meps}

def setLogger(logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.DEBUG)

    path = os.path.join(EXP_DIR, f'{logging_name}.txt')
    if not os.path.exists(path):
        with open(path, 'w') as file:
            file.write('')  
    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def set_args(dataset_name, sens_param_index, sens_param_name, model_name, ):
    opt = {}
    opt['dataset'] = dataset_name
    opt["random_seed"] = 2333
    opt['experiment'] = "LIMI"
    opt["exp_name"] = f"{dataset_name}_{sens_param_name}"
    opt["model_path"] = f'model_info/{model_name}/{dataset_name}_{model_name}_model.pkl'
    # opt["model_path"] = f'model_info/unawareness/{dataset_name}_{sens_param_index}_fakemodel.pkl'
    opt['sens_param'] = sens_param_index + 1
    opt['max_local'] = 100000
    opt['max_global'] = int(opt['max_local'] / 3) + 1
    opt["boundary_file"] = f'model_info/latent_boundary_limi/{dataset_name}/boundary.npy'
    opt['gan_file'] = f'model_info/CTGAN/{dataset_name}_ctgan_model.pkl'
    opt['step'] = 0.3
    opt['expdir'] = os.path.join(
        EXP_DIR,
        'ours' + '_' + str(opt['step']),
        opt["exp_name"],
    )
    if not os.path.exists(opt["expdir"]):
        os.makedirs(opt["expdir"])
    if not os.path.exists('result/LIMI'):
        os.makedirs("result/LIMI")
    opt['global_samples_file'] = os.path.join('result/LIMI', f'{dataset_name}{sens_param_index + 1}_{model_name}_LIMI.npy')
    opt['local_samples_file'] = os.path.join(opt['expdir'], 'local_samples.npy')
    opt['suc_idx_file'] = os.path.join(opt['expdir'], 'suc_idx.npy')
    opt['disc_value_file'] = os.path.join(opt['expdir'], 'disc_value.npy')
    np.random.seed(opt["random_seed"])

    logger = setLogger(opt['exp_name'])
    
    logger.info(f"the random_seed is set as {opt['random_seed']}\n")
    logger.info(f"setting {opt}\n")

    return opt, logger


def numpy2log(arr, output_path):
    f = open(output_path, 'w')
    for item in arr:
        if isinstance(item, list):
            line = ','.join(list(map(str, item)))
        else:
            line = str(item)
        f.write(line + '\n')
    return


def sign(value):
    if value >= 0:
        return 1
    else:
        return -1


class FairTest:
    def __init__(self, opt, logger) -> None:
        self.opt = opt
        self.logger = logger
        self.n_value = 0
        self.global_nvalue = []

        self.build_ctgan_bd_clf()
        self.build_data_and_model()

        self.start_time = time.time()
        self.count = [1]

    def build_data_and_model(self):
        self.samples, self.latent_codes = self.ctgan_syn.sample_and_latent(self.opt['max_global'])
        self.latent_codes = self.latent_codes.detach().cpu().numpy()

        # prepare the testing data and model

        # X, Y, input_shape, nb_classes = dataset_dict[self.opt['dataset']](
        #     self.opt['dataset_path']
        # )
        # self.X, self.Y = X, Y
        # self.input_shape, self.nb_classes = input_shape, nb_classes
        self.data_config = data_config[self.opt['dataset']]

        self.model = joblib.load(self.opt['model_path'])

        self.logger.info(f"load model from {self.opt['model_path']}")

        def calculate_dis(latent_codes, boundary):
            dis_samples = []
            for index in range(0, len(latent_codes)):
                latent_sample = latent_codes[index : index + 1]
                dis_sample = (
                    np.sum(boundary['coef_'] * latent_sample) + boundary['intercept_']
                )
                dis_samples.append(dis_sample[0])
            return np.array(dis_samples)

        dis_samples = calculate_dis(self.latent_codes, self.boundary)
        self.dis_samples = dis_samples

    def load_boundary(self, path: str):
        with open(path, 'rb') as f:
            bd = pickle.load(f)
        ans = bd
        return ans

    def build_ctgan_bd_clf(self):
        self.ctgan_syn : CTGANModel
        self.ctgan_syn = CTGANModel.load(self.opt['gan_file'])
        self.ctgan_syn.set_mode()
        
        self.boundary = self.load_boundary(self.opt['boundary_file'])

    def clip(self, input):
        """
        Clip the generating instance with each feature to make sure it is valid
        :param input: generating instance
        :param conf: the configuration of dataset
        :return: a valid generating instance
        """
        for i in range(len(input)):
            input[i] = max(input[i], self.data_config.input_bounds[i][0])
            input[i] = min(input[i], self.data_config.input_bounds[i][1])
        return input

    def check_for_error_condition(self, t):
        """
        Check whether the test case is an individual discriminatory instance
        :param conf: the configuration of dataset
        :param sess: TF session
        :param x: input placeholder
        :param preds: the model's symbolic output
        :param t: test case
        :param sens: the index of sensitive feature
        :return: whether it is an individual discriminatory instance-0
        """
        t = t.astype("int")
        sens = self.opt['sens_param']
        label = self.model.predict([t])

        # check for all the possible values of sensitive feature
        for val in range(
            self.data_config.input_bounds[sens - 1][0],
            self.data_config.input_bounds[sens - 1][1] + 1,
        ):
            if val != t[sens - 1]:
                tnew = copy.deepcopy(t)
                tnew[sens - 1] = val
                label_new = self.model.predict([tnew])
                if label_new != label:
                    self.n_value = val
                    return True
        return False

    def evaluate_global(self, inp):
        inp = self.clip(inp).astype("int")
        result, real_result = False, False
        temp = copy.deepcopy(inp.astype("int").tolist())
        temp = temp[: self.opt['sens_param'] - 1] + temp[self.opt['sens_param'] :]
        if tuple(temp) not in tot_inputs:
            tot_inputs.add(tuple(temp))
            result = self.check_for_error_condition(inp)

        if result and (tuple(temp) not in global_disc_inputs):
            global_disc_inputs.add(tuple(temp))
            global_disc_inputs_list.append(copy.deepcopy(inp.astype("int").tolist()))
            real_result = True
        return real_result

    def latent_to_data(self, latents):
        fake_data = np.concatenate(latents, axis=0)  
        num_samples = len(fake_data)
        batch_size = self.ctgan_syn._batch_size
        N = int(num_samples / batch_size)
        if num_samples % batch_size != 0:
            N += 1
        data = []

        for ind in tqdm(range(N)):
            fakez = fake_data[ind * batch_size : (ind + 1) * batch_size]
            fakez = torch.Tensor(fakez).cuda()
            fake = self.ctgan_syn._generator(fakez)
            fakeact = self.ctgan_syn._apply_activate(fake, phase="generate")
            data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        raw_data = self.ctgan_syn._transformer.inverse_transform(data).to_numpy()
        return raw_data

    def _count(self, data: np.ndarray):
        a_set = set()
        for item in data.tolist():
            # print(item)
            a_set.add(tuple(item))
        self.logger.info(
            f"item in data is {len(a_set)}, while len(data) is {len(data)}"
        )

    def global_phase_search(self):
        self.logger.info(
            f"iself.opt['max_global'] is {self.opt['max_global']}"
        )
        
        inputs = range(self.opt['max_global'])
        global_latent_list = []

        # 0 1 -1
        self.cnt_candidate = {0: 0, 1: 0, -1: 0}
        candidate_dict = {0: [], 1: [], -1: []}
        step = opt['step']  # 0.3
        for num in tqdm(range(len(inputs))):
            index = inputs[num]
            latent_sample = self.latent_codes[index : index + 1]
            dis_record = self.dis_samples[index]

            sample_zero = latent_sample - dis_record * self.boundary['coef_']
            candidate_dict[0].append(sample_zero)

            # just perturb once
            latent_edit = sample_zero + step * self.boundary['coef_']
            candidate_dict[1].append(latent_edit)
            latent_edit = sample_zero - step * self.boundary['coef_']
            candidate_dict[-1].append(latent_edit)

        candidate_dict_data = {0: [], 1: [], -1: []}
        for perturb in [0, 1, -1]:
            tg_latent_edit_list = candidate_dict[perturb]
            raw_data = self.latent_to_data(latents=tg_latent_edit_list)
            self._count(raw_data)
            candidate_dict_data[perturb] = raw_data

        for ind in tqdm(range(0, len(candidate_dict_data[0]))):
            for perturb in [0, 1, -1]:
                if len(tot_inputs) >= self.opt['max_local']:
                    break
                g_inp_flag = self.evaluate_global(candidate_dict_data[perturb][ind])
                if g_inp_flag:
                    suc_idx.append((ind, perturb))
                    global_latent_list.append(candidate_dict[perturb][ind])
                    self.global_nvalue.append(self.n_value)
                    self.cnt_candidate[perturb] += 1
                    break
            _end = time.time()
            use_time = _end - self.start_time
            sec = len(self.count) * 300
            if use_time >= sec:
                self.logger.info(
                    "Percentage discriminatory inputs - "
                    + str(
                        float(len(global_disc_inputs_list))
                        / float(len(tot_inputs))
                        * 100
                    )
                )
                self.logger.info(
                    "Number of discriminatory inputs are "
                    + str(len(global_disc_inputs_list))
                )
                self.logger.info("Total Inputs are " + str(len(tot_inputs)))

                self.logger.info('use time:' + str(use_time) + "\n")
                self.count.append(1)
            if use_time >= 3600:
                break

        self.logger.info(f"the end ind is {ind}")
        self.tot_global_search = len(tot_inputs)
        IDS = np.array(global_disc_inputs_list)
        np.save(self.opt['global_samples_file'], IDS)
        # numpy2log(global_disc_inputs_list, self.opt['global_samples_file'])
        numpy2log(self.global_nvalue, self.opt['disc_value_file'])

        self.logger.info(
            f"len(global_disc_inputs_list) is {len(global_disc_inputs_list)} in {self.tot_global_search} rate is {len(global_disc_inputs_list)/self.tot_global_search * 100:.2f}"
        )
        self.logger.info(f"self.cnt_candidate is {self.cnt_candidate}")

    def run(self):
        self.global_phase_search()
        self.logger.info("Total Inputs are " + str(len(tot_inputs)))
        self.logger.info(
            f"Total discriminatory inputs of global search- {len(global_disc_inputs)} in {self.opt['max_global']}"
        )
        numpy2log(suc_idx, self.opt['suc_idx_file'])


if __name__ == '__main__':
    # opt, logger = _parse_args()
    dataset_names = ['bank','census', 'credit', 'meps']
    model_name = 'MLP'

    log_path = f'{EXP_DIR}/{model_name}_logging_data.csv'
    log = []
    log_index = []
    column_index = ['NID', "Time", "NPS", "Total Number"]
    
    for dataset_name in dataset_names:
        for sens_param_index in data_config[dataset_name].sensitive_param:
            print(f'###############################  {dataset_name} {sens_param_index}  #####################################')
            log_index.append(f'{dataset_name} {sens_param_index}')
            # store the result of fairness testing
            tot_inputs = set()
            global_disc_inputs = set()
            global_disc_inputs_list = []
            local_disc_inputs = set()
            local_disc_inputs_list = []
            value_list = []
            suc_idx = []
            sens_param_name = data_config[dataset_name].feature_name[sens_param_index]
            opt, logger = set_args(dataset_name, sens_param_index, sens_param_name, model_name)
            fairtest = FairTest(opt, logger)
            start = time.time()
            fairtest.run()
            end = time.time()
            logger.info(f"Total time is {end-start}")
            log.append([len(global_disc_inputs_list), end-start, len(global_disc_inputs_list)/(end - start), len(tot_inputs)])
    log = [[np.round(x, 3) for x in row] for row in log]
    df = pd.DataFrame(log, columns=column_index, index=log_index)
    df.to_csv(log_path, index=True)