from .components import ComponentMeta

aby_millionaire_prob_test_cpn_meta = ComponentMeta("ABYMillionaireProbTest")

@aby_millionaire_prob_test_cpn_meta.bind_param
def aby_millionaire_prob_test_param():
    from federatedml.param.aby_millionaire_prob_test_param import ABYMillionaireProbTestParam

    return ABYMillionaireProbTestParam


@aby_millionaire_prob_test_cpn_meta.bind_runner.on_guest
def aby_millionaire_prob_test_runner_guest():
    from federatedml.ABY.millionaire_prob_test.aby_millionaire_prob_test import ABYMillionaireProbTestGuest

    return ABYMillionaireProbTestGuest


@aby_millionaire_prob_test_cpn_meta.bind_runner.on_host
def aby_millionaire_prob_test_runner_host():
    from federatedml.ABY.millionaire_prob_test.aby_millionaire_prob_test import ABYMillionaireProbTestHost

    return ABYMillionaireProbTestHost