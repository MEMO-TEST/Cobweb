from data_preprocess.config_word import config
from data_preprocess.assertions_df import construct_query_dict, query

Skip_Words = config.Skip_Words
Punctuations = config.Punctuations
country_set = config.country_set
attributes_key = config.attributes_key
sex_set = config.sex_pair_set
query_dict = construct_query_dict('datasets/filtered_assertions.csv')

def selection_with_sensattr(sentence, sens_name):
    word_list = sentence.split()
    sens_index = []
    attr_index = []
    sens_key = country_set if sens_name == 'country' else sex_set
    for i, word in enumerate(word_list):
        if (word not in Skip_Words) and (word not in Punctuations):
            inp_key_list = query(query_dict,word)
            if len(inp_key_list) > 0:
                inp_key = set(inp_key_list)
                if len(inp_key & sens_key) > 0:
                    sens_index.append(i)
                if len(inp_key & attributes_key) > 0:
                    attr_index.append(i)
    if len(sens_index) > 0:
        return 1, sens_index[0]
    elif len(attr_index) > 0:
        return 0, attr_index[0]
    else:
        return -1, -1
