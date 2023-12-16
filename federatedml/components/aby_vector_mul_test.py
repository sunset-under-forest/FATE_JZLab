from .components import ComponentMeta

aby_vector_mul_test_cpn_meta = ComponentMeta("ABYVectorMulTest")

@aby_vector_mul_test_cpn_meta.bind_param
def aby_vector_mul_test_param():
    from federatedml.param.aby_vector_operator_test_param import ABYVectorOperatorParam

    return ABYVectorOperatorParam


@aby_vector_mul_test_cpn_meta.bind_runner.on_guest
def aby_vector_mul_test_runner_guest():
    from federatedml.ABY.vector_operator_test.aby_vector_mul_test import ABYVectorMulTestGuest

    return ABYVectorMulTestGuest


@aby_vector_mul_test_cpn_meta.bind_runner.on_host
def aby_vector_mul_test_runner_host():
    from federatedml.ABY.vector_operator_test.aby_vector_mul_test import ABYVectorMulTestHost

    return ABYVectorMulTestHost
