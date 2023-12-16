from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER

class ABYVectorOperatorParam(BaseParam):
    """
    ABY Vector Operator

    Parameters
    ----------
    opertor : str, supported: "add", "mul", must be in ["add", "mul"], no default value
        Specify the opertor of this party.
    aby_role : str, default: "server"
        Specify the role of this party.
    address : str, default: "0.0.0.0"
        Specify the address of this party.
    port : int, default: 7766
        Specify the port of this party.
    """

    def __init__(self, operator=None, aby_role="server" , address="0.0.0.0", port=7766):
        super(ABYVectorOperatorParam, self).__init__()
        self.operator = None
        self.aby_role = aby_role.lower().strip()
        self.address = address
        self.port = port

    def check(self):
        model_param_descr = "ABY Vector Operator param's "
        if self.operator is not None:
            if not isinstance(self.operator, str):
                raise ValueError(f"{model_param_descr} operator should be str type")
            if self.operator not in ["add", "mul"]:
                raise ValueError(f"{model_param_descr} operator should be 'add' or 'mul'")

            self.operator = self.operator.lower().strip()

        if self.aby_role is not None:
            if not isinstance(self.aby_role, str):
                raise ValueError(f"{model_param_descr} role should be str type")
            if self.aby_role not in ["server", "client"]:
                raise ValueError(f"{model_param_descr} role should be 'server' or 'client'")


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

