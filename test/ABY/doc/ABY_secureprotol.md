# ABY框架加密安全层嵌入开发

承接《ABY框架加法乘法底层算子嵌入》

首先准备好相关环境变量

```BASH
export FEDERATEDML_LIBRARY_PATH=/home/lab/federated_learning/fate/from_src_build/FATE/python/federatedml # {federatedml包路径}
export ABY_LIBRARY_PATH=/home/lab/federated_learning/fate/from_src_build/FATE/python/federatedml/ABY/CPP/extern/ABY # {ABY项目根目录}
export ABY_CPP_SRC_PATH=$FEDERATEDML_LIBRARY_PATH/ABY/CPP
export ABY_COMPONENT_PATH=$FEDERATEDML_LIBRARY_PATH/ABY
export ABY_FATE_TEST_PATH=$FATE_PROJECT_BASE/aby_fate_test
```

## 安全加密层代码编写

```BASH
cd $ABY_COMPONENT_PATH
cp -r ../secureprotol/ aby_secureprotol/
cp -r aby_secureprotol/spdz/ aby_secureprotol/aby
mv aby_secureprotol/aby/spdz.py aby_secureprotol/aby/aby.py
vim aby_secureprotol/aby/aby.py
vim aby_secureprotol/aby/tensor/fixedpoint_numpy.py
vim aby_secureprotol/aby_fixedpoint.py
vim aby_secureprotol/aby/tensor/base.py 
vim aby_secureprotol/aby/__init__.py
```

### aby_secureprotol/aby/aby.py

```PYTHON
from federatedml.ABY.aby_secureprotol.fate_paillier import PaillierKeypair
from federatedml.ABY.aby_secureprotol.aby.communicator import Communicator
from federatedml.ABY.aby_secureprotol.aby.utils import NamingService
from federatedml.ABY.aby_secureprotol.aby.utils import naming


class ABY(object):
    __instance = None

    @classmethod
    def get_instance(cls) -> 'ABY':
        return cls.__instance

    @classmethod
    def set_instance(cls, instance):
        prev = cls.__instance
        cls.__instance = instance
        return prev

    @classmethod
    def has_instance(cls):
        return cls.__instance is not None

    def __init__(self, name="ss", q_field=None, local_party=None, all_parties=None, use_mix_rand=False, n_length=1024):
        self.name_service = naming.NamingService(name)
        self._prev_name_service = None
        self._pre_instance = None

        self.communicator = Communicator(local_party, all_parties)

        self.party_idx = self.communicator.party_idx
        self.other_parties = self.communicator.other_parties
        if len(self.other_parties) > 1:
            raise EnvironmentError("support 2-party secret share only")
        self.public_key, self.private_key = PaillierKeypair.generate_keypair(n_length=n_length)

        if q_field is None:
            q_field = self.public_key.n

        # self.q_field = self._align_q_field(q_field)

        self.use_mix_rand = use_mix_rand

    def __enter__(self):
        self._prev_name_service = NamingService.set_instance(self.name_service)
        self._pre_instance = self.set_instance(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        NamingService.set_instance(self._pre_instance)
        # self.communicator.clean()

    def __reduce__(self):
        raise PermissionError("it's unsafe to transfer this")

    def partial_rescontruct(self):
        # todo: partial parties gets rescontructed tensor
        pass

    @classmethod
    def dot(cls, left, right, target_name=None):
        return left.dot(right, target_name)

    def set_flowid(self, flowid):
        self.communicator.set_flowid(flowid)

    def _align_q_field(self, q_field):
        self.communicator.remote_q_field(q_field=q_field, party=self.other_parties)
        other_q_field = self.communicator.get_q_field(party=self.other_parties)
        other_q_field.append(q_field)
        max_q_field = max(other_q_field)
        return max_q_field

```

### aby_secureprotol/aby/tensor/fixedpoint_numpy.py

```PYTHON
import functools

import numpy as np

from fate_arch.common import Party
from fate_arch.computing import is_table
from federatedml.ABY.aby_secureprotol.aby.beaver_triples import beaver_triplets
from federatedml.ABY.aby_secureprotol.aby.tensor import fixedpoint_table
from federatedml.ABY.aby_secureprotol.aby.tensor.base import ABYTensorBase
from federatedml.ABY.aby_secureprotol.aby.utils import urand_tensor
# from federatedml.ABY.aby_secureprotol.aby.tensor.fixedpoint_endec import FixedPointEndec
from federatedml.ABY.aby_secureprotol.fixedpoint import FixedPointEndec
from federatedml.util import LOGGER
from federatedml.ABY.aby_secureprotol.aby_fixedpoint import ABYFixedPointEndec, ABYFixedPointNumber
from federatedml.ABY.operator.vector_operator64 import vector_add_operator_client, vector_mul_operator_client, \
    vector_add_operator_server, vector_mul_operator_server, \
    vector_operator_execute


class ABYFixedPointTensor(object):
    ...
    address = "127.0.0.1"
    port = 7766

    def __init__(self, value: np.array, endec, name=None):
        if isinstance(value, ABYFixedPointNumber):
            self.value = np.array([value])
        elif isinstance(value, np.ndarray):
            self.value = value
        else:
            raise ValueError(f"type={type(value)}")
        self.endec = endec
        self.name = name

    @property
    def shape(self):
        return self.value.shape

    def reshape(self, shape):
        self.value = self.value.reshape(shape)
        return self

    @property
    def T(self):
        return ABYFixedPointTensor(self.value.T, self.endec)

    def _raw_add(self, other: np.array):
        if isinstance(other, ABYFixedPointTensor):
            other = other.value

        z_value = (self.value + other)
        return ABYFixedPointTensor(z_value, self.endec)

    def _raw_mul(self, other: np.array):
        if isinstance(other, ABYFixedPointTensor):
            other = other.value

        z_value = (self.value * other)
        return ABYFixedPointTensor(z_value, self.endec)

    def __add__(self, other):
        return self._raw_add(other)

    def __mul__(self, other):
        return self._raw_mul(other)

    def __sub__(self, other):
        return self._raw_add(-other)

    @classmethod
    def _vec_share_add(cls, encoded_vector: np.array, role):
        # 判断是一个一维向量
        assert len(encoded_vector.shape) == 1
        assert isinstance(encoded_vector[0], np.uint64)
        vec = encoded_vector

        vec_len = len(vec)
        role = role.lower()
        if role == "server":
            result_vec, result_type = vector_operator_execute(vector_add_operator_server(), vec, cls.address, cls.port)
        elif role == "client":
            result_vec, result_type = vector_operator_execute(vector_add_operator_client(), vec, cls.address, cls.port)
        else:
            raise ValueError(f"unsupported role: {role}")
        result_vector = np.array([x for x in result_vec[:vec_len]])
        return result_vector

    @classmethod
    def _vec_share_mul(cls, encoded_vector: np.array, role):
        # 判断是一个一维向量
        assert len(encoded_vector.shape) == 1
        assert isinstance(encoded_vector[0], np.uint64)
        vec = encoded_vector

        vec_len = len(vec)
        role = role.lower()
        if role == "server":
            result_vec, result_type = vector_operator_execute(vector_mul_operator_server(), vec, cls.address, cls.port)
        elif role == "client":
            result_vec, result_type = vector_operator_execute(vector_mul_operator_client(), vec, cls.address, cls.port)
        else:
            raise ValueError(f"unsupported role: {role}")
        result_vector = np.array([x for x in result_vec[:vec_len]])
        return result_vector

    def share_add(self, role):

        pre_shape = self.shape

        # flat
        encoded_vector = self.value.flatten()
        encoded_vector: np.array = np.array([x.encoding for x in encoded_vector], dtype=np.uint64)
        LOGGER.debug(f"encoded_vector: {encoded_vector} type: {type(encoded_vector)} shape: {encoded_vector.shape}")
        result_vector = self._vec_share_add(encoded_vector, role)
        result_vector = np.array([ABYFixedPointNumber(x) for x in result_vector])
        # reshape
        result_vector = result_vector.reshape(pre_shape)
        return ABYFixedPointTensor(result_vector, self.endec)

    def share_mul(self, role, times=1):
        pre_shape = self.shape

        # flat
        encoded_vector_temp = self.value.flatten()

        encoded_vector_temp: np.array = np.array([x.encoding for x in encoded_vector_temp], dtype=np.uint64)
        encoded_vector = np.array([], dtype=np.uint64)
        # concentrate
        for i in range(times):
            encoded_vector = np.concatenate((encoded_vector, encoded_vector_temp), dtype=np.uint64)

        LOGGER.debug(f"encoded_vector: {encoded_vector} type: {type(encoded_vector)} shape: {encoded_vector.shape}")
        result_vector = self._vec_share_mul(encoded_vector, role)
        # truncate
        result_vector = self.endec.truncate(result_vector)
        # reshape
        if times == 1:
            result_vector = result_vector.reshape(pre_shape)
        else:
            result_vector = result_vector.reshape((pre_shape[0], pre_shape[1] * times))
        return ABYFixedPointTensor(result_vector, self.endec)

    def dot_local(self, other):
        if isinstance(other, ABYFixedPointTensor):
            other = other.value

        ret = np.dot(self.value, other)

        if not isinstance(ret, np.ndarray):
            ret = np.array([ret])

        return ABYFixedPointTensor(ret, self.endec)

    def get(self):
        return self.endec.decode(self.value)


class PaillierFixedPointTensor(ABYTensorBase):
    __array_ufunc__ = None

    def __init__(self, value, tensor_name: str = None, cipher=None):
        super().__init__(q_field=None, tensor_name=tensor_name)
        self.value = value
        self.cipher = cipher

    def dot(self, other, target_name=None):
        def _vec_dot(x, y):
            ret = np.dot(x, y)
            if not isinstance(ret, np.ndarray):
                ret = np.array([ret])
            return ret

        if isinstance(other, (ABYFixedPointTensor, fixedpoint_table.ABYFixedPointTensor)):
            other = other.value
        if isinstance(other, np.ndarray):
            ret = _vec_dot(self.value, other)
            return self._boxed(ret, target_name)
        elif is_table(other):
            f = functools.partial(_vec_dot,
                                  self.value)
            ret = other.mapValues(f)
            return fixedpoint_table.ABYPaillierFixedPointTensor(value=ret,
                                                                tensor_name=target_name,
                                                                cipher=self.cipher)
        else:
            raise ValueError(f"type={type(other)}")

    def broadcast_reconstruct_share(self, tensor_name=None):
        from federatedml.ABY.aby_secureprotol.aby import ABY
        spdz = ABY.get_instance()
        share_val = self.value.copy()
        name = tensor_name or self.tensor_name
        if name is None:
            raise ValueError("name not specified")
        # remote share to other parties
        spdz.communicator.broadcast_rescontruct_share(share_val, name)
        return share_val

    def __str__(self):
        return f"tensor_name={self.tensor_name}, value={self.value}"

    def __repr__(self):
        return self.__str__()

    def _raw_add(self, other):
        z_value = (self.value + other)
        return self._boxed(z_value)

    def _raw_sub(self, other):
        z_value = (self.value - other)
        return self._boxed(z_value)

    def __add__(self, other):
        if isinstance(other, (PaillierFixedPointTensor, ABYFixedPointTensor)):
            return self._raw_add(other.value)
        else:
            return self._raw_add(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (PaillierFixedPointTensor, ABYFixedPointTensor)):
            return self._raw_sub(other.value)
        else:
            return self._raw_sub(other)

    def __rsub__(self, other):
        if isinstance(other, (PaillierFixedPointTensor, ABYFixedPointTensor)):
            z_value = other.value - self.value
        else:
            z_value = other - self.value
        return self._boxed(z_value)

    def __mul__(self, other):
        if isinstance(other, PaillierFixedPointTensor):
            raise NotImplementedError("__mul__ not support PaillierFixedPointTensor")
        elif isinstance(other, ABYFixedPointTensor):
            return self._boxed(self.value * other.value)
        else:
            return self._boxed(self.value * other)

    def __rmul__(self, other):
        self.__mul__(other)

    def _boxed(self, value, tensor_name=None):
        return PaillierFixedPointTensor(value=value,
                                        tensor_name=tensor_name,
                                        cipher=self.cipher)

    @classmethod
    def from_source(cls, tensor_name, source, **kwargs):
        spdz = cls.get_aby()
        q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field
        if 'encoder' in kwargs:
            encoder = kwargs['encoder']
        else:
            base = kwargs['base'] if 'base' in kwargs else 10
            frac = kwargs['frac'] if 'frac' in kwargs else 4
            encoder = FixedPointEndec(n=q_field, field=q_field, base=base, precision_fractional=frac)

        if isinstance(source, np.ndarray):
            _pre = urand_tensor(q_field, source)

            share = _pre

            spdz.communicator.remote_share(share=source - encoder.decode(_pre),
                                           tensor_name=tensor_name,
                                           party=spdz.other_parties[-1])

            return ABYFixedPointTensor(value=share,
                                       q_field=q_field,
                                       endec=encoder,
                                       tensor_name=tensor_name)

        elif isinstance(source, Party):
            share = spdz.communicator.get_share(tensor_name=tensor_name, party=source)[0]

            is_cipher_source = kwargs['is_cipher_source'] if 'is_cipher_source' in kwargs else True
            if is_cipher_source:
                cipher = kwargs['cipher']
                share = cipher.recursive_decrypt(share)
                share = encoder.encode(share)
            return ABYFixedPointTensor(value=share,
                                       q_field=q_field,
                                       endec=encoder,
                                       tensor_name=tensor_name)
        else:
            raise ValueError(f"type={type(source)}")

```

### vim aby_secureprotol/aby_fixedpoint.py

```PYTHON
import functools
import math
import sys

import numpy as np


class ABYFixedPointNumber(object):
    BASE = 2
    TOTAL_BITS = 32
    PRECISION_FRACTIONAL_BITS = 16
    PRECISION_INTEGRAL_BITS = TOTAL_BITS - PRECISION_FRACTIONAL_BITS - 1
    PRECISION_IN_BITS = TOTAL_BITS

    Q = 2 ** PRECISION_IN_BITS

    def __init__(self, value):
        self.value = value

    @property
    def encoding(self):
        return self.value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def precision_from_bits(cls, bits):
        return math.floor(math.log(bits, 2))

    @classmethod
    def encode(cls, rational):
        upscaled = int(rational * cls.BASE ** cls.PRECISION_FRACTIONAL_BITS)
        cls.__overflow_check(upscaled)
        field_element = upscaled % cls.Q
        return cls(field_element)

    def decode(self):
        upscaled = self.value if self.value < self.Q // 2 else self.value - self.Q
        return round(upscaled / self.BASE ** self.PRECISION_FRACTIONAL_BITS,
                     self.precision_from_bits(self.PRECISION_FRACTIONAL_BITS))

    def __add_fixedpointnumber(self, other: "ABYFixedPointNumber"):
        return self.__add_scalar(other.decode())

    def __mul_fixedpointnumber(self, other: "ABYFixedPointNumber"):
        return self.__mul_scalar(other.decode())

    def __add_scalar(self, scalar):
        val = self.decode()
        z = val + scalar
        z_encode = ABYFixedPointNumber.encode(z)
        return z_encode

    def __mul_scalar(self, scalar):
        val = self.decode()
        z = val * scalar
        z_encode = ABYFixedPointNumber.encode(z)
        return z_encode

    def __neg__(self):
        return self.__mul_scalar(-1)

    def __add__(self, other):
        if isinstance(other, ABYFixedPointNumber):
            return self.__add_fixedpointnumber(other)
        else:
            return self.__add_scalar(other)

    def __mul__(self, other):
        if isinstance(other, ABYFixedPointNumber):
            return self.__mul_fixedpointnumber(other)
        else:
            return self.__mul_scalar(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-other)


    @classmethod
    def __overflow_check(cls, value):
        if abs(value) > 2 ** cls.PRECISION_IN_BITS - 1:
            raise ValueError(
                f"Overflow during fixed point encoding: {value} should be in [-{2 ** (cls.PRECISION_INTEGRAL_BITS - 1) - 1}, {2 ** (cls.PRECISION_INTEGRAL_BITS - 1) - 1}]")

    ...


class ABYFixedPointEndec(object):

    def __init__(self):
        ...

    @classmethod
    def _transform_op(cls, tensor, op):

        def _transform(x):
            arr = np.zeros(shape=x.shape, dtype=object)
            view = arr.view().reshape(-1)
            x_array = x.view().reshape(-1)
            for i in range(arr.size):
                view[i] = op(x_array[i])

            return arr

        if isinstance(tensor, (int, np.int16, np.int32, np.int64,
                               float, np.float16, np.float32, np.float64,
                               ABYFixedPointNumber)):
            return op(tensor)

        if isinstance(tensor, np.ndarray):
            z = _transform(tensor)
            return z
        else:
            raise ValueError(f"unsupported type: {type(tensor)}")

    def _encode(self, scalar):
        return ABYFixedPointNumber.encode(scalar)

    def _decode(self, number):
        return number.decode()

    def _truncate(self, number):
        if isinstance(number, ABYFixedPointNumber):
            return ABYFixedPointNumber(number.value // (ABYFixedPointNumber.BASE ** ABYFixedPointNumber.PRECISION_FRACTIONAL_BITS))
        else:
            return ABYFixedPointNumber(number // (ABYFixedPointNumber.BASE ** ABYFixedPointNumber.PRECISION_FRACTIONAL_BITS))

    def encode(self, float_tensor):
        return self._transform_op(float_tensor, op=self._encode)

    def decode(self, integer_tensor):
        if isinstance(integer_tensor, ABYFixedPointNumber):
            return self._decode(integer_tensor)
        elif not isinstance(integer_tensor, np.ndarray):
            raise ValueError(f"unsupported type: {type(integer_tensor)}")
        else:
            return self._transform_op(integer_tensor, op=self._decode)


    def truncate(self, float_tensor):
        return self._transform_op(float_tensor, op=self._truncate)
```

### aby_secureprotol/aby/tensor/base.py 

```PYTHON
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

```

### aby_secureprotol/aby/\_\_init\_\_.py

```PYTHON
from federatedml.ABY.aby_secureprotol.aby.aby import ABY

```

### aby_secureprotol/aby/aby.py

```PYTHON

```

### aby_secureprotol/aby/aby.py

```PYTHON

```

