from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER


class ATestParam(BaseParam):
    """
    TEST

    Parameters
    ----------
    param1 : None or int, default: None
        Specify the random state for shuffle.
    param2 : float or int or None, default: 0.0
        Specify test data set size.
        
    """

    def __init__(self, param1=None, param2=None):
        super(ATestParam, self).__init__()       # super() 函数是用于调用父类(超类)的一个方法。
        self.param1 = param1
        self.param2 = param2

    def check(self):
        model_param_descr = "a test param's "
        if self.param1 is not None:
            if not isinstance(self.param1, int):
                raise ValueError(f"{model_param_descr} param1 should be int type")
            BaseParam.check_nonnegative_number(self.param1, f"{model_param_descr} param1 ")

        if self.param2 is not None:
            BaseParam.check_nonnegative_number(self.param2, f"{model_param_descr} param2 ")
            if isinstance(self.param2, float):
                BaseParam.check_decimal_float(self.param2, f"{model_param_descr} param2 ")
