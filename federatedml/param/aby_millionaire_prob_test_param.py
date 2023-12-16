import random

from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER

class ABYMillionaireProbTestParam(BaseParam):
    """
    ABY Millionaire Prob Test

    Parameters
    ----------
    aby_role : str, default: "server"
        Specify the role of this party.
    money : int, default: a random integer between 1 and 100
        Specify the money of this party.
    address : str, default: "0.0.0.0"
        Specify the address of this party.
    port : int, default: 7766
        Specify the port of this party.
    """

    def __init__(self, aby_role="server" , money=random.randint(1,100), address="0.0.0.0", port=7766):
        super(ABYMillionaireProbTestParam, self).__init__()       # super() 函数是用于调用父类(超类)的一个方法。
        self.aby_role = aby_role.lower().strip()
        self.money = money
        self.address = address
        self.port = port

    def check(self):
        model_param_descr = "ABY Millionaire Prob Test param's "
        if self.aby_role is not None:
            if not isinstance(self.aby_role, str):
                raise ValueError(f"{model_param_descr} role should be str type")
            if self.aby_role not in ["server", "client"]:
                raise ValueError(f"{model_param_descr} role should be 'server' or 'client'")

        if self.money is not None:
            BaseParam.check_nonnegative_number(self.money, f"{model_param_descr} money ")
            if not isinstance(self.money, int):
                raise ValueError(f"{model_param_descr} money should be int type")
            if self.money > 0xffffffff:
                raise ValueError(f"{model_param_descr} money should be less than 0xffffffff")

        if self.address is not None:
            if not isinstance(self.address, str):
                raise ValueError(f"{model_param_descr} address should be str type")
            # 检查是否符合ip地址格式
            import re
            if any(map(lambda n: int(n) > 255, re.match(r'^(\d+)\.(\d+)\.(\d+)\.(\d+)$', self.address).groups())):
                raise ValueError(f"{model_param_descr} address should be ip address format" )

        if self.port is not None:
            if not isinstance(self.port, int):
                raise ValueError(f"{model_param_descr} port should be int type")
            BaseParam.check_positive_integer(self.port, f"{model_param_descr} port ")
            if self.port > 65535:
                raise ValueError(f"{model_param_descr} port should be less than 65535")

