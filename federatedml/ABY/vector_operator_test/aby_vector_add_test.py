from .aby_vector_operator_base import ABYVectorOperator
from federatedml.util import LOGGER
from federatedml.param.aby_vector_operator_test_param import ABYVectorOperatorParam
from federatedml.ABY.operator.vector_operator import vector_add_operator_client, vector_add_operator_server
from ...statistic import data_overview


class ABYVectorAddTest(ABYVectorOperator):
    def __init__(self):
        super().__init__()
        self.model_param = ABYVectorOperatorParam("add")

    def fit(self, train_data, validate_data=None):
        """
        测试
        """
        LOGGER.info("Start ABY Vector Add Test")
        self.fit_prepare(train_data, validate_data)
        self.compute(train_data)


class ABYVectorAddTestGuest(ABYVectorAddTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYVectorAddTestGuest'
        self.set_aby_role("server")
        self.set_operator(vector_add_operator_server())

class ABYVectorAddTestHost(ABYVectorAddTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYVectorAddTestGuest'
        self.set_aby_role("client")
        self.set_operator(vector_add_operator_client())
