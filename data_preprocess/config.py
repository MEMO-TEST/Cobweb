class meps:
    params = 40
    sensitive_param = [2]
    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([0, 85])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 9])
    input_bounds.append([0, 3])
    input_bounds.append([0, 3])
    input_bounds.append([0, 3])
    input_bounds.append([0, 5])
    input_bounds.append([0, 5])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([-9, 70])
    input_bounds.append([-9, 75])
    input_bounds.append([-9, 24])
    input_bounds.append([0, 7])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([0, 2])

    feature_name = ['REGION','AGE','SEX','RACE','MARRY','FTSTU','ACTDTY','HONRDC','RTHLTH',
                    'MNHLTH','CHDDX','ANGIDX','MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON',
                    'CHOLDX','CANCERDX','DIABDX','JTPAIN','ARTHDX','ARTHTYPE','ASTHDX',
                    'ADHDADDX','PREGNT','WLKLIM','ACTLIM','SOCLIM','COGLIM','DFHEAR42',
                    'DFSEE42','ADSMOK42','PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV']
    # the name of each class
    class_name = ["no", "yes"]

    sens_name = {3:'SEX'}

    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    
    discrete_columns_indeces = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39]


class census:
    """
    Configuration of dataset Census Income
    """

    # the size of total features
    params = 13
    sensitive_param = [0, 7, 8]

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 7])
    input_bounds.append([0, 39]) #69 for THEMIS
    input_bounds.append([0, 15])
    input_bounds.append([0, 6])
    input_bounds.append([0, 13])
    input_bounds.append([0, 5])
    input_bounds.append([0, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])
    input_bounds.append([0, 99])
    input_bounds.append([0, 39])

    # the name of each feature
    feature_name = ["Age", "workclass", "fnlwgt", "education", "marital_status", "occupation", "relationship", "Race", "Gender", "capital_gain",
                                                                      "capital_loss", "hours_per_week", "native_country"]

    # the name of each class
    class_name = ["low", "high"]

    sens_name = {9:'Gender',1:"Age",8:"Race"}

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,]
    discrete_columns_indeces = [1,3,4,5,6,7,8,12]

class credit:
    """
    Configuration of dataset German Credit
    """

    # the size of total features
    params = 20
    sensitive_param = [8, 12]

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([0, 3])
    input_bounds.append([1, 80])
    input_bounds.append([0, 4])
    input_bounds.append([0, 10])
    input_bounds.append([1, 200])
    input_bounds.append([0, 4])
    input_bounds.append([0, 4])
    input_bounds.append([1, 4])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 8])
    input_bounds.append([0, 2])
    input_bounds.append([0, 2])
    input_bounds.append([1, 4])
    input_bounds.append([0, 3])
    input_bounds.append([1, 2])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])

    # the name of each feature
    feature_name = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", "savings_status", "employment", "installment_commitment", "Gender", "other_parties",
                                                                      "residence", "property_magnitude", "Age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker"]

    # the name of each class
    class_name = ["bad", "good"]

    sens_name = {9:'Gender',13:"Age"}

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    discrete_columns_indeces = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]

class bank:
    """
    Configuration of dataset Bank Marketing
    """

    # the size of total features
    params = 16
    sensitive_param = [0]

    # the valid religion of each feature
    input_bounds = []
    input_bounds.append([1, 9])
    input_bounds.append([0, 11])
    input_bounds.append([0, 2])
    input_bounds.append([0, 3])
    input_bounds.append([0, 1])
    input_bounds.append([-20, 179])
    input_bounds.append([0, 1])
    input_bounds.append([0, 1])
    input_bounds.append([0, 2])
    input_bounds.append([1, 31])
    input_bounds.append([0, 11])
    input_bounds.append([0, 99])
    input_bounds.append([1, 63])
    input_bounds.append([-1, 39])
    input_bounds.append([0, 1])
    input_bounds.append([0, 3])

    # the name of each feature
    feature_name = ["Age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                                                                      "month", "duration", "campaign", "pdays", "previous", "poutcome"]
    
    # feature_type = ['continuous','discrete','discrete','discrete','discrete','continuous','discrete','discrete','discrete','continuous','continuous','continuous','continuous','continuous','continuous','discrete']
    
    # the name of each class
    class_name = ["no", "yes"]

    sens_name = {1:'Age'}

    # specify the categorical features with their indices
    categorical_features = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    discrete_columns_indeces = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]