from .aby_vector_operator_base import ABYVectorOperator
from federatedml.util import LOGGER
from federatedml.param.aby_vector_operator_test_param import ABYVectorOperatorParam
from federatedml.ABY.operator.vector_operator import vector_mul_operator_server, vector_mul_operator_client
class ABYVectorMulTest(ABYVectorOperator):
    def __init__(self):
        super().__init__()
        self.model_param = ABYVectorOperatorParam("mul")

    def fit(self, train_data, validate_data=None):
        """
        测试
        """
        LOGGER.info("Start ABY Vector Mul Test")
        self.fit_prepare(train_data, validate_data)
        self.compute(train_data)

class ABYVectorMulTestGuest(ABYVectorMulTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYVectorMulTestGuest'
        self.set_aby_role("server")
        self.set_operator(vector_mul_operator_server())

class ABYVectorMulTestHost(ABYVectorMulTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYVectorMulTestHost'
        self.set_aby_role("client")
        self.set_operator(vector_mul_operator_client())
