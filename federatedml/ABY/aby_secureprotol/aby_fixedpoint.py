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