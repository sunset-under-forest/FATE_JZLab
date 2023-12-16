import sys

from federatedml.model_base import ModelBase
from federatedml.param.a_test_param import ATestParam
# ABY LIBRARY
import ctypes

import requests
import traceback

class ATest(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = ATestParam()
        self.model_name = 'ATest'

    def fit(self, *args):
        """
        测试
        """
        port = "8000"
        url = "http://127.0.0.1:" + port + "/"
        response = requests.get(url + "ATest has been called")
        print(response.text)
        # 调用栈
        response = requests.get(url + str(traceback.format_stack()))
        self.dll_test()

        return response.text

    def dll_test(self):

        example = ctypes.CDLL('/tmp/tmp.ZO2yVo1pOb/cmake-build-debug/lib/libmy_dll_test.so')
        result = example.test()






