from .components import ComponentMeta

a_test_cpn_meta = ComponentMeta("ATest")

@a_test_cpn_meta.bind_param
def a_test_param():
    from federatedml.param.a_test_param import ATestParam

    return ATestParam

@a_test_cpn_meta.bind_runner.on_guest.on_local.on_host.on_arbiter
def a_test_runner():
    from federatedml.ABY.test.a_test import ATest
    return ATest

