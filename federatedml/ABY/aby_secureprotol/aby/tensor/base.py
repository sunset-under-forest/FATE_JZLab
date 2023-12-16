import abc

from federatedml.ABY.aby_secureprotol.aby.utils import NamingService


class ABYTensorBase(object):
    __array_ufunc__ = None

    def __init__(self, q_field, tensor_name: str = None):
        self.q_field = q_field

    @classmethod
    def get_aby(cls):
        from federatedml.ABY.aby_secureprotol.aby import ABY
        return ABY.get_instance()

    @abc.abstractmethod
    def dot(self, other, target_name=None):
        pass
