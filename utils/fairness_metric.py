import numpy as np
import joblib
from tqdm import tqdm

class FairnessMeasure():
    def __init__(self):
        pass
    def measure_individual_discrimination(self, model, data_config, sens_index, sample_round, num_gen, scaler=None):
        
        statistics = np.empty(shape=(0, ))
        
        for _ in tqdm(range(sample_round)):
            num_ids = self._generation_random_samples(data_config, model, num_gen, sens_index, scaler)
            percentage = num_ids / num_gen
            statistics = np.append(statistics, [percentage], axis=0)
        avg = np.average(statistics)
        std_dev = np.std(statistics)
        interval = 1.960 * std_dev / np.sqrt(sample_round)
        return f'IFr:{np.round(avg,3)}±{np.round(interval,3)}'
    
    def measure_individual_discrimination_original(self, model, data_config, sens_index, original_data, scaler=None):
        IDS_count = 0
        for data in original_data:
            if self._is_disc_input(data, model, data_config, sens_index, scaler=None):
                IDS_count += 1
        return np.round(IDS_count/len(original_data),3)

    def _generation_random_samples(self, data_config, model, num_gen, sens_index, scaler = None):
        num_attribs = data_config.params
        constraint = data_config.input_bounds
        gen_id = np.empty(shape=(0, num_attribs))
        for i in range(num_gen):
            x_picked = [0] * num_attribs
            for a in range(num_attribs):
                x_picked[a] = np.random.randint(constraint[a][0], constraint[a][1]+1)
            x_picked = np.array(x_picked)
            if scaler != None:
                x_picked = scaler.transform([x_picked])
            if self._is_disc_input(x_picked, model, data_config, sens_index, scaler):
                if x_picked.ndim > 1:
                    gen_id = np.concatenate((gen_id, x_picked), axis=0)
                else:
                    gen_id = np.concatenate((gen_id, [x_picked]), axis=0)
        return len(gen_id)
    
    def _is_disc_input(self, sample, model, data_config, sens_index, scaler=None):
        sens_param_bounds = data_config.input_bounds[sens_index]
        tags = set()
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        elif sample.ndim > 2:
            raise('sample wrong!')
        if model.predict(sample).ndim > 1:
            if scaler == None:
                for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
                    sample[0][sens_index] = sens_value
                    tags.add(np.argmax(model.predict(sample)))
            else:
                for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
                    temp = scaler.inverse_transform(sample)
                    temp[0][sens_index] = sens_value
                    temp = scaler.transform(temp)
                    tags.add(np.argmax(model.predict(temp)))
        else:
            if scaler == None:
                for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
                    sample[0][sens_index] = sens_value
                    tags.add(model.predict(sample)[0])
            else:
                for sens_value in range(sens_param_bounds[0],sens_param_bounds[1]+1):
                    temp = scaler.inverse_transform(sample)
                    temp[0][sens_index] = sens_value
                    temp = scaler.transform(temp)
                    tags.add(model.predict(temp)[0])
        if len(tags) > 1:
            return True
        else:
            return False
    
    def disparate_impact_multi_value_sens(self, model, test_data, sens_index, pos_label=1):
        # when DI > 0.8, the classifier is relatively fair
        predictions = model.predict(test_data)
        if predictions.ndim > 1: 
            predictions = np.argmax(predictions, axis = 1)
        unique_sens_values = np.unique(test_data[:, sens_index])
        disparate_impacts = []
        for sens_value in unique_sens_values:
            sens_attr = test_data[:, sens_index] == sens_value
            p_pos_given_sens = np.mean(predictions[sens_attr] == pos_label)
            p_pos_given_nonsens = np.mean(predictions[~sens_attr] == pos_label)

            disparate_impact = p_pos_given_sens / p_pos_given_nonsens
            disparate_impacts.append(disparate_impact)
        return disparate_impacts
    
    def equal_opportunity_multi_value_sens(self, model, test_data, sens_index, pos_label=1):
        predictions = model.predict(test_data)
        if predictions.ndim > 1: 
            predictions = np.argmax(predictions, axis = 1)
        unique_sens_values = np.unique(test_data[:, sens_index])
        equalities = []
        
        for sens_value in unique_sens_values:
            sens_attr = test_data[:, sens_index] == sens_value
            tp_sens = np.sum(np.logical_and(predictions == pos_label, sens_attr))
            fn_sens = np.sum(np.logical_and(predictions != pos_label, sens_attr))
            tp_nonsens = np.sum(np.logical_and(predictions == pos_label, ~sens_attr))
            fn_nonsens = np.sum(np.logical_and(predictions != pos_label, ~sens_attr))
            
            sens_equality = tp_sens / (tp_sens + fn_sens)
            nonsens_equality = tp_nonsens / (tp_nonsens + fn_nonsens)
            
            equalities.append(sens_equality - nonsens_equality)
    
        return equalities

    def demographic_parity_multi_value_sens(self, model, test_data, sens_index, pos_label=1):
        predictions = model.predict(test_data)
        if predictions.ndim > 1: 
            predictions = np.argmax(predictions, axis = 1)
        unique_sens_values = np.unique(test_data[:, sens_index])
        parities = []
        
        for sens_value in unique_sens_values:
            sens_attr = test_data[:, sens_index] == sens_value
            p_pos_given_sens = np.mean(predictions[sens_attr] == pos_label)
            p_pos_given_nonsens = np.mean(predictions[~sens_attr] == pos_label)
            
            parities.append(abs(p_pos_given_sens - p_pos_given_nonsens))
        
        return parities
    def equalized_odds_multi_value_sens(self, model, test_data, sens_index, pos_label=1):
        predictions = model.predict(test_data)
        if predictions.ndim > 1: 
            predictions = np.argmax(predictions, axis = 1)
        unique_sens_values = np.unique(test_data[:, sens_index])
        equalized_odds = []
        
        for sens_value in unique_sens_values:
            sens_attr = test_data[:, sens_index] == sens_value
            tp_sens = np.sum(np.logical_and(predictions == pos_label, sens_attr))
            fn_sens = np.sum(np.logical_and(predictions != pos_label, sens_attr))
            tn_nonsens = np.sum(np.logical_and(predictions != pos_label, ~sens_attr))
            fp_nonsens = np.sum(np.logical_and(predictions == pos_label, ~sens_attr))
            
            sens_pos_given_sens = tp_sens / (tp_sens + fn_sens)
            sens_pos_given_nonsens = fp_nonsens / (tn_nonsens + fp_nonsens)
            
            equalized_odds.append(abs(sens_pos_given_sens - sens_pos_given_nonsens))
        
        return equalized_odds
# if __name__ == '__main__':
#     dataset_name = 'bank'
#     approach_name = 'expga'
#     data_config = {"census":census, "credit":credit, "bank":bank}
#     fm = FairnessMeasure()
#     model = joblib.load(f'model_info/{dataset_name}_model.pkl')
#     retrain_model = joblib.load(f'retrain_model_info/{dataset_name}_{approach_name}_retrained_model.pkl')
#     fm.measure_individual_discrimination(model, data_config[dataset_name], 100, 10000)
#     print('---------------------------------------------')
#     fm.measure_individual_discrimination(retrain_model, data_config[dataset_name], 100, 10000)
