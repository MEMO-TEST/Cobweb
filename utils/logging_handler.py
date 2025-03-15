import logging
import os

def setLogger(logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f"result/logging_data/{logging_name}.txt")
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger_model_train = setLogger('model_train')
logger_model_retrain = setLogger('model_retrain')
logger_model_test = setLogger('model_test')
logger_model_fm = setLogger('fm')