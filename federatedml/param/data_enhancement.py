from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER


class DataEnhancement(BaseParam):
    """
    数据增强组件参数
    """
    def __init__(self,gamma=1.0,input_dim=10,batch_size=64,epochs=50,eta=0.0002,epsilon_0=1,delta_0=0.00001,c=2,sigma=0.7):
        super().__init__()
        self.gamma = gamma
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = eta
        self.epsilon_0 = epsilon_0
        self.delta_0 = delta_0
        self.c = c
        self.sigma = sigma


    def check(self):
        model_param_descr = "data enhancement param's "
        if self.gamma is not None:
            if not isinstance(self.gamma, float):
                raise ValueError(f"{model_param_descr} gamma should be float type")
            BaseParam.check_positive_number(self.gamma, f"{model_param_descr} gamma ")


