"""
这里的方法是将total_contribution下发给每个客户端，随后客户端在本地计算贡献度，并将贡献度与模型相乘，最后将该模型
发送给服务器，由服务器完成聚合。
"""

from federatedml.framework.homo.blocks import RandomPaddingCipherClient, RandomPaddingCipherServer, PadsCipher, RandomPaddingCipherTransVar
from federatedml.framework.homo.aggregator.aggregator_base import AggregatorBaseClient, AutoSuffix, AggregatorBaseServer
import numpy as np
from federatedml.framework.weights import Weights, NumpyWeights
from federatedml.util import LOGGER
import torch as t
from typing import Union, List
from fate_arch.computing._util import is_table
from federatedml.util import consts


AGG_TYPE = ['weighted_mean', 'sum', 'mean','contribution_mean']


class SecureAggregatorClient(AggregatorBaseClient):

    def __init__(self, secure_aggregate=True, aggregate_type='contribution_mean', aggregate_weight=1.0,individual_contribution = 1.0,
                 total_contribution = 1.0,communicate_match_suffix=None):

        super(SecureAggregatorClient, self).__init__(
            communicate_match_suffix=communicate_match_suffix)
        self.secure_aggregate = secure_aggregate
        self.suffix = {
            "local_loss": AutoSuffix("local_loss"),
            "agg_loss": AutoSuffix("agg_loss"),
            "local_model": AutoSuffix("local_model"),
            "agg_model": AutoSuffix("agg_model"),
            "converge_status": AutoSuffix("converge_status"),
            "total_contribution": AutoSuffix("total_contribution"),
            "individual_contribution": AutoSuffix("individual_contribution") 
         
        }
        #"total_contribution": AutoSuffix("total_contribution")
        # init secure aggregate random padding:
        if self.secure_aggregate:
            self._random_padding_cipher: PadsCipher = RandomPaddingCipherClient(
                trans_var=RandomPaddingCipherTransVar(prefix=communicate_match_suffix)).create_cipher()
            LOGGER.info('initialize secure aggregator done')

        # compute weight
            # 确定权重
        assert aggregate_type in AGG_TYPE, 'aggregate type must be {}'.format(
                AGG_TYPE)
        if aggregate_type == 'weighted_mean':
            LOGGER.info('aggregate type is weighted_mean.')
            aggregate_weight = aggregate_weight
        elif aggregate_type == 'mean':
            LOGGER.info('aggregate type is mean.')
            aggregate_weight = 1
        elif aggregate_type == 'contribution_mean':
            LOGGER.info('aggregate type is contribution_mean.')
            individual_contribution = 1
            total_contribution = 1
            #individual_contribution = self.get_individual_contribution(self.suffix["local_loss"])

        self.send(aggregate_weight, suffix=('agg_weight',))
        LOGGER.info('Client send aggregate weight done.')
        
        self.send(total_contribution,suffix=('total_contribution',))
        LOGGER.info('Client send total_contribution done.')
        
        self.send(individual_contribution,suffix=('individual_contribution',))
        LOGGER.info('Client send individual_contribution done.')
        #total_contribution = self.get_total_contribution(self.suffix["total_contribution"])
        self._weight = individual_contribution / total_contribution  
        LOGGER.info('Client compute weight done.')
        # individual contribution / total contribution
        """
        self._weight = aggregate_weight / \
            self.get(suffix=('agg_weight', ))[0]
            """
        if aggregate_type == 'sum':  # reset _weight
            self._weight = 1

        self._set_table_amplify_factor = False

        LOGGER.debug('aggregate compute weight is {}'.format(self._weight))
    
    def get_contribution(self, individual_contribution, total_contribution):
        LOGGER.debug('Client get_contribution beggin')
        contribution = individual_contribution / total_contribution
        LOGGER.debug('Client get_contribution done, value is {}'.format(contribution))
        return contribution

    def _process_model(self, model):
        
        LOGGER.info('Client process model begin.')
        # self._weight = self.get_contribution(individual_contribution, total_contribution)
        LOGGER.info('Client process weight is {}.'.format(self._weight))
        
        to_agg = None

        if isinstance(model, np.ndarray) or isinstance(model, Weights):
            if isinstance(model, np.ndarray):
                to_agg = NumpyWeights(model * self._weight)
            else:
                to_agg = model * self._weight

            if self.secure_aggregate:
                to_agg: Weights = to_agg.encrypted(
                    self._random_padding_cipher)
            LOGGER.info('Client process model done.')
            return to_agg

        # is FATE distrubed Table
        elif is_table(model):
            model = model.mapValues(lambda x: x * self._weight)

            if self.secure_aggregate:
                if not self._set_table_amplify_factor:
                    self._random_padding_cipher.set_amplify_factor(
                        consts.SECURE_AGG_AMPLIFY_FACTOR)
                model = self._random_padding_cipher.encrypt_table(model)
            LOGGER.info('Client process model done.')
            return model

        if isinstance(model, t.nn.Module):
            parameters = list(model.parameters())
            tmp_list = [[np.array(p.cpu().detach().tolist()) for p in parameters if p.requires_grad]]
            LOGGER.debug('Aggregate trainable parameters: {}/{}'.format(len(tmp_list[0]), len(parameters)))
        elif isinstance(model, t.optim.Optimizer):
            tmp_list = [[np.array(p.cpu().detach().tolist()) for p in group["params"]]
                        for group in model.param_groups]
        elif isinstance(model, list):
            for p in model:
                assert isinstance(
                    p, np.ndarray), 'expecting List[np.ndarray], but got {}'.format(p)
            tmp_list = [model]

        if self.secure_aggregate:
            to_agg = [
                [
                    NumpyWeights(
                        arr *
                        self._weight).encrypted(
                        self._random_padding_cipher) for arr in arr_list] for arr_list in tmp_list]
        else:
            to_agg = [[arr * self._weight for arr in arr_list]
                      for arr_list in tmp_list]
        LOGGER.info('Client process model done.')
        return to_agg

    def get_individual_contribution(self, local_loss):

        LOGGER.info('Client get_individual_contribution bigin.')
        if local_loss != 0:
            individual_contribution = 1.0 / local_loss
        else:
            individual_contribution = 1
        LOGGER.info('Client get_individual_contribution done.')

        return individual_contribution
    
    def _recover_model(self, model, agg_model):

        if isinstance(model, np.ndarray):
            return agg_model.unboxed
        elif isinstance(model, Weights):
            return agg_model
        elif is_table(agg_model):
            return agg_model
        else:
            if self.secure_aggregate:
                agg_model = [[np_weight.unboxed for np_weight in arr_list]
                             for arr_list in agg_model]

            if isinstance(model, t.nn.Module):
                for agg_p, p in zip(agg_model[0], [p for p in model.parameters() if p.requires_grad]):
                    p.data.copy_(t.Tensor(agg_p))

                return model
            elif isinstance(model, t.optim.Optimizer):
                for agg_group, group in zip(agg_model, model.param_groups):
                    for agg_p, p in zip(agg_group, group["params"]):
                        p.data.copy_(t.Tensor(agg_p))
                return model
            else:
                return agg_model

    def send_loss(self, loss, suffix=tuple()):
        suffix = self._get_suffix('local_loss', suffix)
        assert isinstance(loss, float) or isinstance(
            loss, np.ndarray), 'illegal loss type {}, loss should be a float or a np array'.format(type(loss))
        self.send(loss * self._weight, suffix)

    def send_model(self,
                   model: Union[np.ndarray,
                                Weights,
                                List[np.ndarray],
                                t.nn.Module,
                                t.optim.Optimizer],
                   suffix=tuple(),
                   individual_contribution=1.0
                   ):
        """Sending model to arbiter for aggregation

        Parameters
        ----------
        model : model can be:
                A numpy array
                A Weight instance(or subclass of Weights), see federatedml.framework.weights
                List of numpy array
                A pytorch model, is the subclass of torch.nn.Module
                A pytorch optimizer, will extract param group from this optimizer as weights to aggregate
        suffix : sending suffix, by default tuple(), can be None or tuple contains str&number. If None, will automatically generate suffix
        """
        suffix = self._get_suffix('local_model', suffix)
        suffix1 = tuple()
        suffix1 = self._get_suffix('individual_contribution',suffix1)
        self.send(individual_contribution,suffix1)
        
        # judge model type
        to_agg_model = self._process_model(model)
        self.send(to_agg_model, suffix)

    def send_individual_contribution(self, individual_contribution, suffix=tuple()):
        suffix = self._get_suffix('individual_contribution', suffix)
        self.send(individual_contribution, suffix)
        
    def get_aggregated_model(self, suffix=tuple()):
        LOGGER.info('Client get_aggregated_model.')
        suffix = self._get_suffix("agg_model", suffix)
        LOGGER.info('Client get_aggregated_model success.')
        LOGGER.info('Client total_contribution begin.')
        suffix1 = tuple()
        suffix1 = self._get_suffix("total_contribution", suffix1)
        
        return self.get(suffix)[0], self.get(suffix1)[0]

    def get_aggregated_loss(self, suffix=tuple()):
        LOGGER.info('Client get_aggregated_loss.')
        suffix = self._get_suffix("agg_loss", suffix)
        return self.get(suffix)[0]

    def get_total_contribution(self, individual_contribution, suffix = tuple()):
        self.send_individual_contribution(individual_contribution, suffix = suffix)
        LOGGER.info('Client get_total_contribution.')
        suffix1 = tuple()
        suffix1 = self._get_suffix("total_contribution", suffix1)
        return self.get(suffix)[0]

    def get_converge_status(self, suffix=tuple()):
        LOGGER.info('Client get_converge_status.')
        suffix = self._get_suffix("converge_status", suffix)
        return self.get(suffix)[0]

    def model_aggregation(self, model,individual_contribution=1.0, suffix=tuple()):
        LOGGER.info('Client get_individual_contribution value is {}.'.format(individual_contribution))
        #suffix1 = tuple()
        #total_contribution = self.get_converge_status(suffix1)
        #LOGGER.info('Client get_total_contribution value is {}.'.format(total_contribution))
        self.send_model(model, suffix=suffix, individual_contribution=individual_contribution)
        agg_model, total_contribution = self.get_aggregated_model(suffix=suffix)
        LOGGER.info('Client get_total_contribution success.total_contribution = {}'.format(total_contribution))
        LOGGER.info('Client get_aggregated_model done')
        self._weight = self.get_contribution(individual_contribution=individual_contribution, total_contribution=total_contribution)
        
        return self._recover_model(model, agg_model)

    def loss_aggregation(self, loss, suffix=tuple()):
        self.send_loss(loss, suffix=suffix)
        #self._weight = self.get_individual_contribution(loss) / self.get_total_contribution()
        converge_status = self.get_converge_status(suffix=suffix)
        return converge_status


class SecureAggregatorServer(AggregatorBaseServer):

    def __init__(self, secure_aggregate=True, communicate_match_suffix=None):
        super(SecureAggregatorServer, self).__init__(
            communicate_match_suffix=communicate_match_suffix)
        self.suffix = {
            "local_loss": AutoSuffix("local_loss"),
            "agg_loss": AutoSuffix("agg_loss"),
            "local_model": AutoSuffix("local_model"),
            "agg_model": AutoSuffix("agg_model"),
            "converge_status": AutoSuffix("converge_status"),
            "total_contribution": AutoSuffix("total_contribution"),
            "individual_contribution": AutoSuffix("individual_contribution") 
            
        }
        #"total_contribution": AutoSuffix("total_contribution")
        self.secure_aggregate = secure_aggregate
        if self.secure_aggregate:
            RandomPaddingCipherServer(trans_var=RandomPaddingCipherTransVar(
                prefix=communicate_match_suffix)).exchange_secret_keys()
            LOGGER. info('initialize secure aggregator done')

        agg_weights = self.collect(suffix=('agg_weight', )) 
        total_contribution = self.collect(suffix=('total_contribution',))
        sum_weights = 0
        sum_total_contribution = 0
        for i in total_contribution:
            sum_total_contribution +=i
        for i in agg_weights:
            sum_weights += i
        self.broadcast(sum_weights, suffix=('agg_weight', ))
        self.broadcast(sum_total_contribution, suffix=('total_contribution', ))
        LOGGER. info('initialize secure aggregator done') 

    def get_total_contribution(self, contribution_list):
        """
        用于计算客户端贡献度

        Args:
            suffix (tuple, optional): 元组，用于存储后缀为“local_loss”的值. 默认为空.
            party_idx (int, optional): 整数，下标值。默认为-1.

        Returns:
            client_contribution: 客户端贡献度
        """
        #suffix = self._get_suffix('individual_contribution', suffix)
        #contribution_list = self.collect(suffix, party_idx=party_idx)

        # 计算1 / L，并计算总和
        #inverse_losses = (1.0 / loss for loss in losses if loss != 0)
        total_contribution = contribution_list[0]
        for con in contribution_list[1:]:
            total_contribution += con
        #total_contribution = losses
        return total_contribution
    
    def aggregate_model(self, suffix=None, party_idx=-1):
        suffix1 = tuple()
        suffix1 = self._get_suffix('individual_contribution', suffix1)
        total_contribution = self.collect(suffix = suffix1, party_idx=party_idx)
        LOGGER.info('server  collect individual contribution : {}'.format(total_contribution))
        LOGGER.info('server  aggregate_model')
        # get suffix
        suffix = self._get_suffix('local_model', suffix)
        LOGGER.info('server  get local_model')
        # recv params for aggregation
        models = self.collect(suffix=suffix, party_idx=party_idx)
        LOGGER.info('models is')
        agg_result = None

        # Aggregate Weights or Numpy Array
        if isinstance(models[0], Weights):
            agg_result = models[0]
            for w in models[1:]:
                agg_result += w
            LOGGER.info('local_model is numpy')

        # Aggregate Table
        elif is_table(models[0]):
            agg_result = models[0]
            for table in models[1:]:
                agg_result = agg_result.join(table, lambda x1, x2: x1 + x2)
            LOGGER.info('local_model is table')
            return agg_result, total_contribution

        # Aggregate numpy groups
        elif isinstance(models[0], list):
            # aggregation
            agg_result = models[0]
            # aggregate numpy model weights from all clients
            for params_group in models[1:]:
                for agg_params, params in zip(
                        agg_result, params_group):
                    for agg_p, p in zip(agg_params, params):
                        # agg_p: NumpyWeights or numpy array
                        agg_p += p
            LOGGER.info('local_model is list')

        if agg_result is None:
            raise ValueError(
                'can not aggregate receive model, format is illegal: {}'.format(models))

        return agg_result, total_contribution

    def broadcast_model(self, model, total_contribution, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('agg_model', suffix)
        LOGGER.info('Server broadcast_model')
        self.broadcast(model, suffix=suffix, party_idx=party_idx)
        LOGGER.info('Server broadcast_model done.')
        suffix1 = tuple()
        suffix1 = self._get_suffix('total_contribution', suffix1)
        self.broadcast(total_contribution, suffix=suffix1, party_idx=party_idx)

    def aggregate_loss(self, suffix=tuple(), party_idx=-1):

        # get loss
        suffix = self._get_suffix('local_loss', suffix)
        losses = self.collect(suffix, party_idx=party_idx)
        # aggregate loss
        total_loss = losses[0]
        for loss in losses[1:]:
            total_loss += loss

        return total_loss

    def broadcast_loss(self, loss_sum, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('agg_loss', suffix)
        self.broadcast(loss_sum, suffix=suffix, party_idx=party_idx)

    def model_aggregation(self, suffix=tuple(), party_idx=-1):
        agg_model ,total_contribution1 = self.aggregate_model(suffix=suffix, party_idx=party_idx)
        total_contribution = self.get_total_contribution(total_contribution1)
        LOGGER.info('Server aggregate_model done, and get total_contribution = {}'.format(total_contribution))
        self.broadcast_model(agg_model, total_contribution, suffix=suffix, party_idx=party_idx)
        #suffix1 =tuple()
        #suffix1 = self._get_suffix('individual_contribution',suffix1)
        #total_contribution = self.get_total_contribution(suffix=suffix1,party_idx=party_idx)
        LOGGER.info('Server broadcast_total_contribution done')
        #self.broadcast_total_contribution(total_contribution,suffix=suffix1,party_idx=party_idx)
        #LOGGER.info('Server broadcast_total_contribution done')

        return agg_model

    def broadcast_converge_status(self, converge_status, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('converge_status', suffix)
        self.broadcast(converge_status, suffix=suffix, party_idx=party_idx)

    def broadcast_total_contribution(self, total_contribution, suffix=tuple(), party_idx=-1):
        suffix = self._get_suffix('total_contribution', suffix)
        self.broadcast(total_contribution, suffix=suffix, party_idx=party_idx)

    def loss_aggregation(self, check_converge=False, converge_func=None, suffix=tuple(), party_idx=-1):
        agg_loss = self.aggregate_loss(suffix=suffix, party_idx=party_idx)
        if check_converge:
            converge_status = converge_func(agg_loss)
        else:
            converge_status = False
        LOGGER.info('mmmmmm')
        self.broadcast_converge_status(
            converge_status, suffix=suffix, party_idx=party_idx)
        LOGGER.info('!!!!!!!!!!')
        return agg_loss, converge_status
