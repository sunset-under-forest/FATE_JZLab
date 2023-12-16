import numpy as np

from federatedml.model_base import ModelBase
from federatedml.ABY.operator.vector_operator import vector_operator_dll,vector_operator_execute
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from typing import List

class ABYVectorOperator(ModelBase):
    def __init__(self):
        super().__init__()
        self.dll = vector_operator_dll
        self.aby_role = None
        self.operator = None
        self.result_vector = None
        self.vector = None
        self.vector_size = None


    def fit_prepare(self, train_data, validate_data=None):
        """
        测试
        """
        aby_role = self.model_param.aby_role
        LOGGER.info("aby_role: {}".format(aby_role))
        if aby_role!= self.aby_role:
            raise ValueError("aby_role should be {}" .format(self.aby_role))
        address = self.model_param.address
        port = self.model_param.port
        LOGGER.info("address: {}".format(address))
        LOGGER.info("port: {}".format(port))
        LOGGER.debug("dll: {}".format(self.dll))
        LOGGER.debug("train_data: {}".format(train_data))
        LOGGER.debug("validate_data: {}".format(validate_data))
        LOGGER.debug("operator: {}".format(self.operator))
        temp = train_data.first()
        LOGGER.info("TTTTTTTTT {}".format(temp))
        LOGGER.info("TTTTTTTTT {}".format(type(temp)))
        LOGGER.info("TTTTTTTTT {}".format(temp[1].__dict__))
        LOGGER.info("TTTTTTTTT {}".format(temp[1].features))
        LOGGER.info("TTTTTTTTT {}".format(train_data.count()))
        LOGGER.info("TTTTTTTTT {}".format(train_data._table))
        # LOGGER.info("TTTTTTTTT {}".format(list(train_data.collect())))
        self.vector_size = train_data.count()

        # for v in (list(train_data.collect())):
        #     LOGGER.info("id:{}, value:{}".format(v[0], v[1].features))

        self.vector = np.array([v[1].features[0] for v in train_data.collect()])

        # LOGGER.info("TTTTTTTTT {}".format(self.vector))


    def compute(self, vector:List[int]):
        LOGGER.info("Start ABY Vector Operator Compute")

        vector = np.array(self.vector).astype(np.int32)
        LOGGER.info("vector: {}".format(vector))

        result_vector, return_type =  vector_operator_execute(self.operator,  vector, self.model_param.address, self.model_param.port)

        # retype result_vector to List[int]
        LOGGER.info("result_vector: {}".format(result_vector))
        LOGGER.info("return_type: {}".format(return_type))

        self.result_vector = result_vector[:self.vector_size]
        LOGGER.info("self.result_vector: {}".format(self.result_vector))

        LOGGER.info("Finish ABY Vector Operator Compute")

    def set_aby_role(self, role):
        self.aby_role = role.lower().strip()

    def set_operator(self, operator:callable):
        self.operator = operator