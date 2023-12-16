from federatedml.model_base import ModelBase
from federatedml.param.aby_millionaire_prob_test_param import ABYMillionaireProbTestParam
from federatedml.util import LOGGER
import ctypes
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ABY_PKG_PATH = os.path.join(CURRENT_PATH, "..")
DLL_PATH = os.path.join(ABY_PKG_PATH, "lib")


class ABYMillionaireProbTest(ModelBase):
    def __init__(self):
        super().__init__()
        self.dll = None
        self.load_dll()

    def load_dll(self):
        self.dll = ctypes.CDLL(os.path.join(DLL_PATH, "libFATE_ABY_millionaire_prob_test_lib.so"))


class ABYMillionaireProbTestGuest(ABYMillionaireProbTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYMillionaireProbTestGuest'
        self.model_param = ABYMillionaireProbTestParam()

    def fit(self, train_data, validate_data=None):
        """
        测试
        """
        LOGGER.info("Start ABY Millionaire Prob Test Guest")
        money = self.model_param.money
        aby_role = self.model_param.aby_role
        LOGGER.info("aby_role: {}".format(aby_role))
        if aby_role!= "server":
            raise ValueError("aby_role should be server as bob")
        LOGGER.info("So this is BOB and BOB's money: {}".format(money))
        address = self.model_param.address
        port = self.model_param.port
        LOGGER.info("address: {}".format(address))
        LOGGER.info("port: {}".format(port))
        LOGGER.debug("dll: {}".format(self.dll))
        result = self.dll.bob(money, address.encode(), port)

        # 这里的百万富翁例子是判断ALICE的钱是否大于BOB的钱，也就是说如果result为1，说明ALICE的钱比BOB多，为0则相反
        LOGGER.info("result: {}".format(result))
        if result == 1:
            LOGGER.info("ALICE is richer than BOB")
        elif result == 0:
            LOGGER.info("BOB is richer than ALICE")

        return result


class ABYMillionaireProbTestHost(ABYMillionaireProbTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYMillionaireProbTestHost'
        self.model_param = ABYMillionaireProbTestParam()

    def fit(self, train_data, validate_data=None):
        """
        测试
        """
        LOGGER.info("Start ABY Millionaire Prob Test Host")
        money = self.model_param.money
        aby_role = self.model_param.aby_role
        LOGGER.info("aby_role: {}".format(aby_role))
        if aby_role!= "client":
            raise ValueError("aby_role should be client as alice")
        LOGGER.info("So this is ALICE and ALICE's money: {}".format(money))
        address = self.model_param.address
        port = self.model_param.port
        LOGGER.info("address: {}".format(address))
        LOGGER.info("port: {}".format(port))
        LOGGER.debug("dll: {}".format(self.dll))
        result = self.dll.alice(money, address.encode(), port)

        # 这里的百万富翁例子是判断ALICE的钱是否大于BOB的钱，也就是说如果result为1，说明ALICE的钱比BOB多，为0则相反
        LOGGER.info("result: {}".format(result))
        if result == 1:
            LOGGER.info("ALICE is richer than BOB")
        elif result == 0:
            LOGGER.info("BOB is richer than ALICE")
        return result

