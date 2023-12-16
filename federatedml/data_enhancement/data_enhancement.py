from federatedml.model_base import ModelBase
from federatedml.util import LOGGER

class DataEnhancement(ModelBase):
    """
    数据增强组件
    """
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def _init_model(self, param):
        self.gamma = param.gamma

    def _load_model(self, model_dict):
        self.gamma = model_dict["gamma"]

    def _save_model(self):
        return {"gamma": self.gamma}

    def fit(self, data):
        LOGGER.info("Start data enhancement fit ...")



        return data

