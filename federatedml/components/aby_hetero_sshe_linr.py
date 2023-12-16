from .components import ComponentMeta

aby_hetero_sshe_linr_cpn_meta = ComponentMeta("ABYHeteroSSHELinR")


@aby_hetero_sshe_linr_cpn_meta.bind_param
def aby_hetero_sshe_linr_param():
    from federatedml.param.aby_hetero_sshe_linr_param import ABYHeteroSSHELinRParam

    return ABYHeteroSSHELinRParam


@aby_hetero_sshe_linr_cpn_meta.bind_runner.on_guest
def aby_hetero_sshe_linr_runner_guest():
    from federatedml.ABY.linear_model.bilateral_linear_model.hetero_sshe_linear_regression.hetero_linr_guest import (
        ABYHeteroLinRGuest,
    )

    return ABYHeteroLinRGuest


@aby_hetero_sshe_linr_cpn_meta.bind_runner.on_host
def aby_hetero_sshe_linr_runner_host():
    from federatedml.ABY.linear_model.bilateral_linear_model.hetero_sshe_linear_regression.hetero_linr_host import (
        ABYHeteroLinRHost,
    )

    return ABYHeteroLinRHost