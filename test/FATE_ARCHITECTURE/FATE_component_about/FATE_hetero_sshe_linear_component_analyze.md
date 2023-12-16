# hetero_sshe_linear组件分析

`HeteroLinRGuest`来自`federatedml.linear_model.bilateral_linear_model.hetero_sshe_linear_regression.hetero_linr_guest`

![federatedml.linear_model.bilateral_linear_model.hetero_sshe_linear_regression.hetero_linr_guest](.\images\federatedml.linear_model.bilateral_linear_model.hetero_sshe_linear_regression.hetero_linr_guest.png)

## 训练过程入口点

### HeteroLinRGuest.fit

```PYTHON
    def fit(self, data_instances, validate_data=None):
        LOGGER.info("Starting to fit hetero_sshe_linear_regression")
        self.prepare_fit(data_instances, validate_data)

        self.fit_single_model(data_instances, validate_data)
```

### HeteroSSHEBase.fit_single_model

```PYTHON
    def fit_single_model(self, data_instances, validate_data=None):
        LOGGER.info(f"Start to train single {self.model_name}")
        if len(self.component_properties.host_party_idlist) > 1:
            raise ValueError(f"Hetero SSHE Model does not support multi-host training.")
        self.callback_list.on_train_begin(data_instances, validate_data)

        model_shape = self.get_features_shape(data_instances)
        instances_count = data_instances.count()

        if not self.component_properties.is_warm_start:
            w = self._init_weights(model_shape)
            self.model_weights = LinearModelWeights(l=w,
                                                    fit_intercept=self.model_param.init_param.fit_intercept)
            last_models = copy.deepcopy(self.model_weights)
        else:
            last_models = copy.deepcopy(self.model_weights)
            w = last_models.unboxed
            self.callback_warm_start_init_iter(self.n_iter_)

        if self.role == consts.GUEST:
            if with_weight(data_instances):
                LOGGER.info(f"data with sample weight, use sample weight.")
                if self.model_param.early_stop == "diff":
                    LOGGER.warning("input data with weight, please use 'weight_diff' for 'early_stop'.")
                data_instances = scale_sample_weight(data_instances)
        self.batch_generator.initialize_batch_generator(data_instances, batch_size=self.batch_size)

        with SPDZ(
            "hetero_sshe",
            local_party=self.local_party,
            all_parties=self.parties,
            q_field=self.q_field,
            use_mix_rand=self.model_param.use_mix_rand,
        ) as spdz:
            spdz.set_flowid(self.flowid)
            self.secure_matrix_obj.set_flowid(self.flowid)
            # not sharing the model when reveal_every_iter
            if not self.reveal_every_iter:
                w_self, w_remote = self.share_model(w, suffix="init")
                last_w_self, last_w_remote = w_self, w_remote
                LOGGER.debug(f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")
            batch_data_generator = self.batch_generator.generate_batch_data()

            encoded_batch_data = []
            batch_labels_list = []
            batch_weight_list = []

            for batch_data in batch_data_generator:
                if self.fit_intercept:
                    batch_features = batch_data.mapValues(lambda x: np.hstack((x.features, 1.0)))
                else:
                    batch_features = batch_data.mapValues(lambda x: x.features)
                if self.role == consts.GUEST:
                    batch_labels = batch_data.mapValues(lambda x: np.array([x.label], dtype=self.label_type))
                    batch_labels_list.append(batch_labels)
                    if self.weight:
                        batch_weight = batch_data.mapValues(lambda x: np.array([x.weight], dtype=float))
                        batch_weight_list.append(batch_weight)
                    else:
                        batch_weight_list.append(None)

                self.batch_num.append(batch_data.count())

                encoded_batch_data.append(
                    fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(batch_features),
                                                      q_field=self.fixedpoint_encoder.n,
                                                      endec=self.fixedpoint_encoder))

            while self.n_iter_ < self.max_iter:
                self.callback_list.on_epoch_begin(self.n_iter_)
                LOGGER.info(f"start to n_iter: {self.n_iter_}")

                loss_list = []

                self.optimizer.set_iters(self.n_iter_)
                if not self.reveal_every_iter:
                    self.self_optimizer.set_iters(self.n_iter_)
                    self.remote_optimizer.set_iters(self.n_iter_)

                for batch_idx, batch_data in enumerate(encoded_batch_data):
                    current_suffix = (str(self.n_iter_), str(batch_idx))
                    if self.role == consts.GUEST:
                        batch_labels = batch_labels_list[batch_idx]
                        batch_weight = batch_weight_list[batch_idx]
                    else:
                        batch_labels = None
                        batch_weight = None

                    if self.reveal_every_iter:
                        y = self.forward(weights=self.model_weights,
                                         features=batch_data,
                                         labels=batch_labels,
                                         suffix=current_suffix,
                                         cipher=self.cipher,
                                         batch_weight=batch_weight)
                    else:
                        y = self.forward(weights=(w_self, w_remote),
                                         features=batch_data,
                                         labels=batch_labels,
                                         suffix=current_suffix,
                                         cipher=self.cipher,
                                         batch_weight=batch_weight)

                    if self.role == consts.GUEST:
                        if self.weight:
                            error = y - batch_labels.join(batch_weight, lambda y, b: y * b)
                        else:
                            error = y - batch_labels

                        self_g, remote_g = self.backward(error=error,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher)
                    else:
                        self_g, remote_g = self.backward(error=y,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher)

                    # loss computing;
                    suffix = ("loss",) + current_suffix
                    if self.reveal_every_iter:
                        batch_loss = self.compute_loss(weights=self.model_weights,
                                                       labels=batch_labels,
                                                       suffix=suffix,
                                                       cipher=self.cipher)
                    else:
                        batch_loss = self.compute_loss(weights=(w_self, w_remote),
                                                       labels=batch_labels,
                                                       suffix=suffix,
                                                       cipher=self.cipher)

                    if batch_loss is not None:
                        batch_loss = batch_loss * self.batch_num[batch_idx]
                    loss_list.append(batch_loss)

                    if self.reveal_every_iter:
                        # LOGGER.debug(f"before reveal: self_g shape: {self_g.shape}, remote_g_shape: {remote_g}，"
                        #              f"self_g: {self_g}")

                        new_g = self.reveal_models(self_g, remote_g, suffix=current_suffix)

                        # LOGGER.debug(f"after reveal: new_g shape: {new_g.shape}, new_g: {new_g}"
                        #              f"self.model_param.reveal_strategy: {self.model_param.reveal_strategy}")

                        if new_g is not None:
                            self.model_weights = self.optimizer.update_model(self.model_weights, new_g,
                                                                             has_applied=False)

                        else:
                            self.model_weights = LinearModelWeights(
                                l=np.zeros(self_g.shape),
                                fit_intercept=self.model_param.init_param.fit_intercept)
                    else:
                        if self.optimizer.penalty == consts.L2_PENALTY:
                            self_g = self_g + self.self_optimizer.alpha * w_self
                            remote_g = remote_g + self.remote_optimizer.alpha * w_remote

                        # LOGGER.debug(f"before optimizer: {self_g}, {remote_g}")

                        self_g = self.self_optimizer.apply_gradients(self_g)
                        remote_g = self.remote_optimizer.apply_gradients(remote_g)

                        # LOGGER.debug(f"after optimizer: {self_g}, {remote_g}")
                        w_self -= self_g
                        w_remote -= remote_g

                        LOGGER.debug(f"w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

                if self.role == consts.GUEST:
                    loss = np.sum(loss_list) / instances_count
                    self.loss_history.append(loss)
                    if self.need_call_back_loss:
                        self.callback_loss(self.n_iter_, loss)
                else:
                    loss = None

                if self.converge_func_name in ["diff", "abs"]:
                    self.is_converged = self.check_converge_by_loss(loss, suffix=(str(self.n_iter_),))
                elif self.converge_func_name == "weight_diff":
                    if self.reveal_every_iter:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=last_models.unboxed,
                            new_w=self.model_weights.unboxed,
                            suffix=(str(self.n_iter_),))
                        last_models = copy.deepcopy(self.model_weights)
                    else:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=(last_w_self, last_w_remote),
                            new_w=(w_self, w_remote),
                            suffix=(str(self.n_iter_),))
                        last_w_self, last_w_remote = copy.deepcopy(w_self), copy.deepcopy(w_remote)
                else:
                    raise ValueError(f"Cannot recognize early_stop function: {self.converge_func_name}")

                LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
                self.callback_list.on_epoch_end(self.n_iter_)
                self.n_iter_ += 1

                if self.stop_training:
                    break

                if self.is_converged:
                    break

            # Finally reconstruct
            if not self.reveal_every_iter:
                new_w = self.reveal_models(w_self, w_remote, suffix=("final",))
                if new_w is not None:
                    self.model_weights = LinearModelWeights(
                        l=new_w,
                        fit_intercept=self.model_param.init_param.fit_intercept)

        LOGGER.debug(f"loss_history: {self.loss_history}")
        self.set_summary(self.get_model_summary())

```

## 权重初始化

权重初始化部分，经过调试已知`self.component_properties.is_warm_start`默认是false

```PYTHON
        if not self.component_properties.is_warm_start:
            w = self._init_weights(model_shape)
            self.model_weights = LinearModelWeights(l=w,
                                                    fit_intercept=self.model_param.init_param.fit_intercept)
            LOGGER.debug(f"self.model_weights: {self.model_weights}")


            last_models = copy.deepcopy(self.model_weights)
```

HeteroSSHEBase._init_weights

```PYTHON
    def _init_weights(self, model_shape):
        return self.initializer.init_model(model_shape, init_params=self.init_param_obj)
```

Initializer.init_model

```PYTHON
    def init_model(self, model_shape, init_params, data_instance=None):
        init_method = init_params.init_method
        fit_intercept = init_params.fit_intercept

        random_seed = init_params.random_seed
        np.random.seed(random_seed)

        if fit_intercept:
            if isinstance(model_shape, int):
                model_shape += 1
            else:
                new_shape = []
                for ds in model_shape:
                    new_shape.append(ds + 1)
                model_shape = tuple(new_shape)

        if init_method == 'random_normal':
            w = self.random_normal(model_shape)
        elif init_method == 'random_uniform':
            w = self.random_uniform(model_shape)
        elif init_method == 'ones':
            w = self.ones(model_shape)
        elif init_method == 'zeros':
            w = self.zeros(model_shape, fit_intercept, data_instance)
        elif init_method == 'const':
            init_const = init_params.init_const
            w = self.constant(model_shape, const=init_const)
        else:
            raise NotImplementedError("Initial method cannot be recognized: {}".format(init_method))
        # LOGGER.debug("Inited model is :{}".format(w))
        return w

```

Initializer.random_normal

```PYTHON
    def random_normal(self, data_shape):
        if isinstance(data_shape, Iterable):
            inits = np.random.randn(*data_shape)
        else:
            inits = np.random.randn(data_shape)
        return inits
```

权重`w`是一个numpy数组，维度由跟`data_shape`决定，等于特征维度，权重随机生成，随机方法包括正太，均匀等，只要固定了随机数种子`random_seed`，就可以固定每一次的权重`w`，`random_seed`来自`self.init_param_obj`

`self.init_param_obj`在`BaseLinearModel`定义，在`_init_model`重写方法中赋值。

```PYTHON
    def _init_model(self, params):
        self.model_param = params
        self.alpha = params.alpha
        self.init_param_obj = params.init_param
        ...
```

`_init_model`方法在`ModelBase._run`中被调用

```PYTHON
    def _run(self, cpn_input) -> None:
        # paramters
        self.model_param.update(cpn_input.parameters)
        self.model_param.check()
        self.component_properties.parse_component_param(
            cpn_input.roles, self.model_param
        )
        self.role = self.component_properties.role
        self.component_properties.parse_dsl_args(cpn_input.datasets, cpn_input.models)
        self.component_properties.parse_caches(cpn_input.caches)
        self.anonymous_generator = Anonymous(role=self.role, party_id=self.component_properties.local_partyid)
        # init component, implemented by subclasses
        self._init_model(self.model_param)
        ...
```

此处的`self.model_param`就是在子类中需要定义的参数对象，是`BaseParam`的子类对象，或者说实现了`BaseParam`接口。

HeteroLinRGuest

```PYTHON
class HeteroLinRGuest(HeteroSSHEGuestBase):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLinearRegression'
        self.model_param_name = 'HeteroLinearRegressionParam'
        self.model_meta_name = 'HeteroLinearRegressionMeta'
        self.model_param = HeteroSSHELinRParam()
```

HeteroSSHELinRParam.\_\_init\_\_

```PYTHON
    def __init__(self, penalty='L2',
                 tol=1e-4, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=20, early_stop='diff',
```

所以`self.init_param_obj`默认等于`InitParam()`

于是`random_seed`默认等于`InitParam()`的`random_seed`

InitParam.\_\_init\_\_

```PYTHON
    def __init__(self, init_method='random_uniform', init_const=1, fit_intercept=True, random_seed=None):
        super().__init__()
        self.init_method = init_method
        self.init_const = init_const
        self.fit_intercept = fit_intercept
        self.random_seed = random_seed
```

`random_seed`默认为None，所以每一次都是随机的。

现在我们的目的是照着FATE框架的hetero_sshe_linear组件写一个以ABY为底层安全协议的hetero_sshe_linear组件，数据角度来说，预期最好就是从原始训练数据，训练过程每一步的权重，梯度变化，直至最后训练出来的模型是完全一致的，不同的只有训练过程使用的底层安全多方计算支持，前者是SPDZ协议（FATE原生）实现，后者（我们现在要做的）是ABY框架实现。

我们要固定住初始化的权重，所以要设置`random_seed`的值。

但是现在问题是`random_seed`所处的位置是参数中的参数，在发布作业时，可以通过conf文件来设置参数，但是参数中的参数要怎么设置呢？

### 设置`random_seed`

首先定位到参数赋值的地方

ModelBase._run

```PYTHON
    def _run(self, cpn_input) -> None:
        # paramters
        self.model_param.update(cpn_input.parameters)
        self.model_param.check()
        self.component_properties.parse_component_param(
            cpn_input.roles, self.model_param
        )
```

BaseParam.update

```python
    def update(self, conf, allow_redundant=False):
        return ParamExtract().recursive_parse_param_from_config(
            param=self,
            config_json=conf,
            param_parse_depth=0,
            valid_check=not allow_redundant,
            name=self._name,
        )
```

ParamExtract.recursive_parse_param_from_config

```PYTHON
    def recursive_parse_param_from_config(
        self, param, config_json, param_parse_depth, valid_check, name
    ):
        if param_parse_depth > PARAM_MAXDEPTH:
            raise ValueError("Param define nesting too deep!!!, can not parse it")

        inst_variables = param.__dict__ # 获取参数类的所有属性

        for variable in inst_variables:
            attr = getattr(param, variable)

            if type(attr).__name__ in self.builtin_types or attr is None:   # 如果属性是内置类型或者为空
                if variable in config_json: # 如果属性在配置文件中
                    option = config_json[variable]  # 获取配置文件中的属性值
                    setattr(param, variable, option)    # 设置属性值
            elif variable in config_json:   # 如果属性不是内置类型，且在配置文件中
                sub_params = self.recursive_parse_param_from_config(    # 递归调用
                    attr,
                    config_json.get(variable),
                    param_parse_depth + 1,
                    valid_check,
                    name,
                )
                setattr(param, variable, sub_params)

        if valid_check:
            redundant = []
            for var in config_json:
                if var not in inst_variables:
                    redundant.append(var)

            if redundant:
                raise ValueError(f"cpn `{name}` has redundant parameters {redundant}")

        return param
```

所以可以修改`hetero_linr_conf.json`如下

```JSON
{
    "dsl_version": 2,
    "initiator": {
        "role": "guest",
        "party_id": 10000
    },
    "role": {
        "guest": [
            10000
        ],
        "host": [
            9999
        ]
    },
    "job_parameters": {
        "common": {
            "job_type": "train"
        }
    },
    "component_parameters": {
        "role": {
            "host": {
                "0": {
                    "data_transform_0": {
                        "with_label": false
                    },
                    "reader_0": {
                        "table": {
                            "name": "my_linr_test_host",
                            "namespace": "test"
                        }
                    }
                }
            },
            "guest": {
                "0": {
                    "data_transform_0": {
                        "with_label": true,
                        "label_name": "y",
                        "label_type": "float",
                        "output_format": "dense"
                    },
                    "reader_0": {
                        "table": {
                            "name": "my_linr_test_guest",
                            "namespace": "test"
                        }
                    }
                }
            }
        },
        "common": {
            "hetero_linr_0": {
                "penalty": "L2",
                "tol": 0.001,
                "alpha": 0.01,
                "optimizer": "sgd",
                "batch_size": -1,
                "learning_rate": 0.15,
                "init_param": {
                    "init_method": "zeros"
                },
                "max_iter": 20,
                "early_stop": "weight_diff",
                "decay": 0.0,
                "decay_sqrt": false,
                "reveal_every_iter": true
            },
            "evaluation_0": {
                "eval_type": "regression",
                "pos_label": 1
            },
            "feature_scale_0": {
                "method": "min_max_scale",
                "mode": "cap",
                "feat_upper": 1,
                "feat_lower": 0
            }
        }
    }
}
```

其中

```JSON
            "hetero_linr_0": {
                "penalty": "L2",
                "tol": 0.001,
                "alpha": 0.01,
                "optimizer": "sgd",
                "batch_size": -1,
                "learning_rate": 0.15,
                "init_param": {
                    "init_method": "zeros"
                },
                "max_iter": 20,
                "early_stop": "weight_diff",
                "decay": 0.0,
                "decay_sqrt": false,
                "reveal_every_iter": true
            },
```

init_param再使用大括号嵌套，即可将参数的参数属性传入，这里默认初始化参数`w`为全零数组

接着使用`w`创建`LinearModelWeights`对象，之后通过属性`self.model_weights`访问权重

```PYTHON
            self.model_weights = LinearModelWeights(l=w,                    fit_intercept=self.model_param.init_param.fit_intercept)

```

(自己加的DEBUG记录)

```
[21740:139823571107840] - [hetero_sshe_linear_model.fit_single_model] [line:231]: model_shape: 1
[DEBUG] [2023-11-08 15:50:45,222] [202311081550260538370] [21740:139823571107840] - [hetero_sshe_linear_model.fit_single_model] [line:233]: instances_count: 100
[DEBUG] [2023-11-08 15:50:45,223] [202311081550260538370] [21740:139823571107840] - [hetero_sshe_linear_model.fit_single_model] [line:237]: w: [0. 0.]
[DEBUG] [2023-11-08 15:50:45,223] [202311081550260538370] [21740:139823571107840] - [hetero_sshe_linear_model.fit_single_model] [line:240]: self.model_weights: weights: [0.], intercept: 0.0
```

## 准备训练`batch`

### 创建`self.batch_generator`

承接上面的代码

```PYTHON
        if self.role == consts.GUEST:
            if with_weight(data_instances):
                LOGGER.info(f"data with sample weight, use sample weight.")
                if self.model_param.early_stop == "diff":
                    LOGGER.warning("input data with weight, please use 'weight_diff' for 'early_stop'.")
                data_instances = scale_sample_weight(data_instances)
        self.batch_generator.initialize_batch_generator(data_instances, batch_size=self.batch_size)
```

if条件就是判断数据是否带着权重，然后将纯训练特征提取出来。重点关注`self.batch_generator`的操作

创建`self.batch_generator`的过程

#### HeteroSSHEGuestBase.prepare_fit

```python
    def prepare_fit(self, data_instances, validate_data):
        # self.transfer_variable = SSHEModelTransferVariable()
        self.batch_generator = batch_generator.Guest()
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)
        self.header = copy.deepcopy(data_instances.schema.get("header", []))
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)
```

`prepare_fit`在`fit_single_model`之前被调用。

#### batch_generator.Guest

```PYTHON
class Guest(batch_info_sync.Guest):
    def __init__(self):
        self.mini_batch_obj = None
        self.finish_sycn = False
        self.batch_nums = None
        self.batch_masked = False

    def register_batch_generator(self, transfer_variables, has_arbiter=True):
        self._register_batch_data_index_transfer(transfer_variables.batch_info,
                                                 transfer_variables.batch_data_index,
                                                 getattr(transfer_variables, "batch_validate_info", None),
                                                 has_arbiter)
	...
```

就是一些训练`batch`信息的初始化和赋值

#### batch_generator.Guest.initialize_batch_generator

```PYTHON
    def initialize_batch_generator(self, data_instances, batch_size, suffix=tuple(),
                                   shuffle=False, batch_strategy="full", masked_rate=0):
        self.mini_batch_obj = MiniBatch(data_instances, batch_size=batch_size, shuffle=shuffle,
                                        batch_strategy=batch_strategy, masked_rate=masked_rate)
        self.batch_nums = self.mini_batch_obj.batch_nums
        self.batch_masked = self.mini_batch_obj.batch_size != self.mini_batch_obj.masked_batch_size
        batch_info = {"batch_size": self.mini_batch_obj.batch_size, "batch_num": self.batch_nums,
                      "batch_mutable": self.mini_batch_obj.batch_mutable,
                      "masked_batch_size": self.mini_batch_obj.masked_batch_size}
        self.sync_batch_info(batch_info, suffix)

        if not self.mini_batch_obj.batch_mutable:
            self.prepare_batch_data(suffix)
```

#### self.mini_batch_obj

##### MiniBatch

```python
class MiniBatch:
    def __init__(self, data_inst, batch_size=320, shuffle=False, batch_strategy="full", masked_rate=0):
        self.batch_data_sids = None
        self.batch_nums = 0
        self.data_inst = data_inst
        self.all_batch_data = None
        self.all_index_data = None
        self.data_sids_iter = None
        self.batch_data_generator = None
        self.batch_mutable = False
        self.batch_masked = False

        if batch_size == -1:
            self.batch_size = data_inst.count()
        else:
            self.batch_size = batch_size

        self.__init_mini_batch_data_seperator(data_inst, self.batch_size, batch_strategy, masked_rate, shuffle)
        
    def __init_mini_batch_data_seperator(self, data_insts, batch_size, batch_strategy, masked_rate, shuffle):
        self.data_sids_iter, data_size = indices.collect_index(data_insts)

        self.batch_data_generator = get_batch_generator(
            data_size, batch_size, batch_strategy, masked_rate, shuffle=shuffle)
        self.batch_nums = self.batch_data_generator.batch_nums
        self.batch_mutable = self.batch_data_generator.batch_mutable()
        self.masked_batch_size = self.batch_data_generator.masked_batch_size

        if self.batch_mutable is False:
            self.__generate_batch_data()
            
def get_batch_generator(data_size, batch_size, batch_strategy, masked_rate, shuffle):
    if batch_size >= data_size:
        LOGGER.warning("As batch_size >= data size, all batch strategy will be disabled")
        return FullBatchDataGenerator(data_size, data_size, shuffle=False)

    # if round((masked_rate + 1) * batch_size) >= data_size:
        # LOGGER.warning("Masked dataset's batch_size >= data size, batch shuffle will be disabled")
        # return FullBatchDataGenerator(data_size, data_size, shuffle=False, masked_rate=masked_rate)
    if batch_strategy == "full":
        if masked_rate > 0:
            LOGGER.warning("If using full batch strategy and masked rate > 0, shuffle will always be true")
            shuffle = True
        return FullBatchDataGenerator(data_size, batch_size, shuffle=shuffle, masked_rate=masked_rate)
    else:
        if shuffle:
            LOGGER.warning("if use random select batch strategy, shuffle will not work")
        return RandomBatchDataGenerator(data_size, batch_size, masked_rate)

```

##### FullBatchDataGenerator

```PYTHON
class FullBatchDataGenerator(BatchDataGenerator):
    def __init__(self, data_size, batch_size, shuffle=False, masked_rate=0):
        super(FullBatchDataGenerator, self).__init__(data_size, batch_size, shuffle, masked_rate=masked_rate)
        self.batch_nums = (data_size + batch_size - 1) // batch_size

        LOGGER.debug(f"Init Full Batch Data Generator, batch_nums: {self.batch_nums}, batch_size: {self.batch_size}, "
                     f"masked_batch_size: {self.masked_batch_size}, shuffle: {self.shuffle}")
```

##### BatchDataGenerator

```PYTHON
class BatchDataGenerator(object):
    def __init__(self, data_size, batch_size, shuffle=False, masked_rate=0):
        self.batch_nums = None
        self.masked_batch_size = min(data_size, round((1 + masked_rate) * batch_size))
        self.batch_size = batch_size
        self.shuffle = shuffle

```

结合日志可知`self.mini_batch_obj`是一个`FullBatchDataGenerator`对象，`self.batch_nums`在此例中为1，`self.batch_masked`为True，`self.mini_batch_obj.batch_mutable`为False，之后会通过`self.sync_batch_info`方法跟其他参与方同步生成的训练批次信息，之后也会调用`self.prepare_batch_data`方法。

```
[WARNING] [2023-11-08 15:50:45,248] [202311081550260538370] [21740:139823571107840] - [mini_batch.get_batch_generator] [line:92]: As batch_size >= data size, all batch strategy will be disabled
[DEBUG] [2023-11-08 15:50:45,248] [202311081550260538370] [21740:139823571107840] - [mini_batch.__init__] [line:140]: Init Full Batch Data Generator, batch_nums: 1, batch_size: 100, masked_batch_size: 100, shuffle: False
...
[mini_batch.mini_batch_data_generator] [line:56]: Currently, batch_num is: 1
...
```

### 根据`self.batch_generator`准备好每个`batch`的数据

```python
    with SPDZ(
        "hetero_sshe",
        local_party=self.local_party,
        all_parties=self.parties,
        q_field=self.q_field,
        use_mix_rand=self.model_param.use_mix_rand,
    ) as spdz:
        spdz.set_flowid(self.flowid)
        self.secure_matrix_obj.set_flowid(self.flowid)
        # not sharing the model when reveal_every_iter
        if not self.reveal_every_iter:
            w_self, w_remote = self.share_model(w, suffix="init")
            last_w_self, last_w_remote = w_self, w_remote
            LOGGER.debug(f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")
        batch_data_generator = self.batch_generator.generate_batch_data()

        encoded_batch_data = []
        batch_labels_list = []
        batch_weight_list = []

        for batch_data in batch_data_generator:
            if self.fit_intercept:
                batch_features = batch_data.mapValues(lambda x: np.hstack((x.features, 1.0)))
            else:
                batch_features = batch_data.mapValues(lambda x: x.features)
            if self.role == consts.GUEST:
                batch_labels = batch_data.mapValues(lambda x: np.array([x.label], dtype=self.label_type))
                batch_labels_list.append(batch_labels)
                if self.weight:
                    batch_weight = batch_data.mapValues(lambda x: np.array([x.weight], dtype=float))
                    batch_weight_list.append(batch_weight)
                else:
                    batch_weight_list.append(None)

            self.batch_num.append(batch_data.count())

            encoded_batch_data.append(
                fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(batch_features),
                                                  q_field=self.fixedpoint_encoder.n,
                                                  endec=self.fixedpoint_encoder))
```

```
[DEBUG] [2023-11-08 17:54:55,262] [202311081754382799510] [23627:140419722838016] - [mini_batch.mini_batch_data_generator] [line:56]: Currently, batch_num is: 1
[DEBUG] [2023-11-08 17:54:55,356] [202311081754382799510] [23627:140419722838016] - [hetero_sshe_linear_model.fit_single_model] [line:298]: encoded_batch_data: [tensor_name=41b23f045f1bd2a8a2f61c6dcbdc1c5c, value=<fate_arch.computing.standalone._table.Table object at 0x7fb5e2de79d0>]
[DEBUG] [2023-11-08 17:54:55,356] [202311081754382799510] [23627:140419722838016] - [hetero_sshe_linear_model.fit_single_model] [line:299]: batch_labels_list: [<fate_arch.computing.standalone._table.Table object at 0x7fb5e3047610>]
[DEBUG] [2023-11-08 17:54:55,356] [202311081754382799510] [23627:140419722838016] - [hetero_sshe_linear_model.fit_single_model] [line:300]: batch_weight_list: [None]
```

## 训练过程

`self.reveal_every_iter`定义在`conf`文件中，默认为true。

```PYTHON

            while self.n_iter_ < self.max_iter:
                self.callback_list.on_epoch_begin(self.n_iter_)
                LOGGER.info(f"start to n_iter: {self.n_iter_}")

                loss_list = []

                self.optimizer.set_iters(self.n_iter_)
                if not self.reveal_every_iter:
                    self.self_optimizer.set_iters(self.n_iter_)
                    self.remote_optimizer.set_iters(self.n_iter_)

                for batch_idx, batch_data in enumerate(encoded_batch_data):
                    current_suffix = (str(self.n_iter_), str(batch_idx))
                    if self.role == consts.GUEST:
                        batch_labels = batch_labels_list[batch_idx]
                        batch_weight = batch_weight_list[batch_idx]
                    else:
                        batch_labels = None
                        batch_weight = None

                    if self.reveal_every_iter:
                        y = self.forward(weights=self.model_weights,
                                         features=batch_data,
                                         labels=batch_labels,
                                         suffix=current_suffix,
                                         cipher=self.cipher,
                                         batch_weight=batch_weight)
                    else:
                        y = self.forward(weights=(w_self, w_remote),
                                         features=batch_data,
                                         labels=batch_labels,
                                         suffix=current_suffix,
                                         cipher=self.cipher,
                                         batch_weight=batch_weight)

                    if self.role == consts.GUEST:
                        if self.weight:
                            error = y - batch_labels.join(batch_weight, lambda y, b: y * b)
                        else:
                            error = y - batch_labels

                        self_g, remote_g = self.backward(error=error,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher)
                    else:
                        self_g, remote_g = self.backward(error=y,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher)

                    # loss computing;
                    suffix = ("loss",) + current_suffix
                    if self.reveal_every_iter:
                        batch_loss = self.compute_loss(weights=self.model_weights,
                                                       labels=batch_labels,
                                                       suffix=suffix,
                                                       cipher=self.cipher)
                    else:
                        batch_loss = self.compute_loss(weights=(w_self, w_remote),
                                                       labels=batch_labels,
                                                       suffix=suffix,
                                                       cipher=self.cipher)

                    if batch_loss is not None:
                        batch_loss = batch_loss * self.batch_num[batch_idx]
                    loss_list.append(batch_loss)

                    if self.reveal_every_iter:
                        # LOGGER.debug(f"before reveal: self_g shape: {self_g.shape}, remote_g_shape: {remote_g}，"
                        #              f"self_g: {self_g}")

                        new_g = self.reveal_models(self_g, remote_g, suffix=current_suffix)

                        # LOGGER.debug(f"after reveal: new_g shape: {new_g.shape}, new_g: {new_g}"
                        #              f"self.model_param.reveal_strategy: {self.model_param.reveal_strategy}")

                        if new_g is not None:
                            self.model_weights = self.optimizer.update_model(self.model_weights, new_g,
                                                                             has_applied=False)

                        else:
                            self.model_weights = LinearModelWeights(
                                l=np.zeros(self_g.shape),
                                fit_intercept=self.model_param.init_param.fit_intercept)
                    else:
                        if self.optimizer.penalty == consts.L2_PENALTY:
                            self_g = self_g + self.self_optimizer.alpha * w_self
                            remote_g = remote_g + self.remote_optimizer.alpha * w_remote

                        # LOGGER.debug(f"before optimizer: {self_g}, {remote_g}")

                        self_g = self.self_optimizer.apply_gradients(self_g)
                        remote_g = self.remote_optimizer.apply_gradients(remote_g)

                        # LOGGER.debug(f"after optimizer: {self_g}, {remote_g}")
                        w_self -= self_g
                        w_remote -= remote_g

                        LOGGER.debug(f"w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

                if self.role == consts.GUEST:
                    loss = np.sum(loss_list) / instances_count
                    self.loss_history.append(loss)
                    if self.need_call_back_loss:
                        self.callback_loss(self.n_iter_, loss)
                else:
                    loss = None

                if self.converge_func_name in ["diff", "abs"]:
                    self.is_converged = self.check_converge_by_loss(loss, suffix=(str(self.n_iter_),))
                elif self.converge_func_name == "weight_diff":
                    if self.reveal_every_iter:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=last_models.unboxed,
                            new_w=self.model_weights.unboxed,
                            suffix=(str(self.n_iter_),))
                        last_models = copy.deepcopy(self.model_weights)
                    else:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=(last_w_self, last_w_remote),
                            new_w=(w_self, w_remote),
                            suffix=(str(self.n_iter_),))
                        last_w_self, last_w_remote = copy.deepcopy(w_self), copy.deepcopy(w_remote)
                else:
                    raise ValueError(f"Cannot recognize early_stop function: {self.converge_func_name}")

                LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
                self.callback_list.on_epoch_end(self.n_iter_)
                self.n_iter_ += 1

                if self.stop_training:
                    break

                if self.is_converged:
                    break

            # Finally reconstruct
            if not self.reveal_every_iter:
                new_w = self.reveal_models(w_self, w_remote, suffix=("final",))
                if new_w is not None:
                    self.model_weights = LinearModelWeights(
                        l=new_w,
                        fit_intercept=self.model_param.init_param.fit_intercept)

```

### 前向

关注到

`batch_weight`默认为None

```PYTHON
                    if self.reveal_every_iter:
                        y = self.forward(weights=self.model_weights,
                                         features=batch_data,	# 
                                         labels=batch_labels,
                                         suffix=current_suffix,
                                         cipher=self.cipher,
                                         batch_weight=batch_weight)
```

#### HeteroLinRGuest.forward

```PYTHON
    def forward(self, weights, features, labels, suffix, cipher, batch_weight):
        self._cal_z(weights, features, suffix, cipher)
        complete_z = self.wx_self + self.wx_remote  # complete_z = z_guest + z_host = w_guest * x_guest + w_host * x_host = w * x

        self.encrypted_wx = complete_z

        self.encrypted_error = complete_z - labels  # encrypted_error = z - y = w * x - y
        if batch_weight:
            complete_z = complete_z * batch_weight
            self.encrypted_error = self.encrypted_error * batch_weight

        tensor_name = ".".join(("complete_z",) + suffix)
        shared_z = SecureMatrix.from_source(tensor_name,
                                            complete_z,
                                            cipher,
                                            self.fixedpoint_encoder.n,
                                            self.fixedpoint_encoder)    # return the MPC result to every party
        return shared_z

```

##### HeteroSSHEGuestBase._cal_z

```PYTHON
    def _cal_z(self, weights, features, suffix, cipher):
        if not self.reveal_every_iter:
            LOGGER.info(f"[forward]: Calculate z in share...")
            w_self, w_remote = weights
            z = self._cal_z_in_share(w_self, w_remote, features, suffix, cipher)
        else:
            LOGGER.info(f"[forward]: Calculate z directly...")
            w = weights.unboxed
            z = features.dot_local(w)

        remote_z = self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                                 is_remote=False,
                                                                 cipher=None,
                                                                 z=None)[0]

        self.wx_self = z
        self.wx_remote = remote_z
```

`z = w * x`

`complete_z = z_guest + z_host = w_guest * x_guest + w_host * x_host = w * x`

`encrypted_error = z - y = w * x - y`

### 反向

```PYTHON
                    if self.role == consts.GUEST:
                        if self.weight:
                            error = y - batch_labels.join(batch_weight, lambda y, b: y * b)
                        else:
                            error = y - batch_labels

                        self_g, remote_g = self.backward(error=error,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher)
                    else:
                        self_g, remote_g = self.backward(error=y,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher)

```

### 计算损失

```python
                    # loss computing;
                    suffix = ("loss",) + current_suffix
                    if self.reveal_every_iter:
                        batch_loss = self.compute_loss(weights=self.model_weights,
                                                       labels=batch_labels,
                                                       suffix=suffix,
                                                       cipher=self.cipher)
                    else:
                        batch_loss = self.compute_loss(weights=(w_self, w_remote),
                                                       labels=batch_labels,
                                                       suffix=suffix,
                                                       cipher=self.cipher)

                    if batch_loss is not None:
                        batch_loss = batch_loss * self.batch_num[batch_idx]
                    loss_list.append(batch_loss)

```

### 梯度更新

```python
                    if self.reveal_every_iter:
                        # LOGGER.debug(f"before reveal: self_g shape: {self_g.shape}, remote_g_shape: {remote_g}，"
                        #              f"self_g: {self_g}")

                        new_g = self.reveal_models(self_g, remote_g, suffix=current_suffix)

                        # LOGGER.debug(f"after reveal: new_g shape: {new_g.shape}, new_g: {new_g}"
                        #              f"self.model_param.reveal_strategy: {self.model_param.reveal_strategy}")

                        if new_g is not None:
                            self.model_weights = self.optimizer.update_model(self.model_weights, new_g,
                                                                             has_applied=False)

                        else:
                            self.model_weights = LinearModelWeights(
                                l=np.zeros(self_g.shape),
                                fit_intercept=self.model_param.init_param.fit_intercept)
                    else:
                        if self.optimizer.penalty == consts.L2_PENALTY:
                            self_g = self_g + self.self_optimizer.alpha * w_self
                            remote_g = remote_g + self.remote_optimizer.alpha * w_remote

                        # LOGGER.debug(f"before optimizer: {self_g}, {remote_g}")

                        self_g = self.self_optimizer.apply_gradients(self_g)
                        remote_g = self.remote_optimizer.apply_gradients(remote_g)

                        # LOGGER.debug(f"after optimizer: {self_g}, {remote_g}")
                        w_self -= self_g
                        w_remote -= remote_g

                        LOGGER.debug(f"w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")
```

### 判断是否收敛

```PYTHON
                if self.role == consts.GUEST:
                    loss = np.sum(loss_list) / instances_count
                    self.loss_history.append(loss)
                    if self.need_call_back_loss:
                        self.callback_loss(self.n_iter_, loss)
                else:
                    loss = None

                if self.converge_func_name in ["diff", "abs"]:
                    self.is_converged = self.check_converge_by_loss(loss, suffix=(str(self.n_iter_),))
                elif self.converge_func_name == "weight_diff":
                    if self.reveal_every_iter:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=last_models.unboxed,
                            new_w=self.model_weights.unboxed,
                            suffix=(str(self.n_iter_),))
                        last_models = copy.deepcopy(self.model_weights)
                    else:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=(last_w_self, last_w_remote),
                            new_w=(w_self, w_remote),
                            suffix=(str(self.n_iter_),))
                        last_w_self, last_w_remote = copy.deepcopy(w_self), copy.deepcopy(w_remote)
                else:
                    raise ValueError(f"Cannot recognize early_stop function: {self.converge_func_name}")

                LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
                self.callback_list.on_epoch_end(self.n_iter_)
                self.n_iter_ += 1

                if self.stop_training:
                    break

                if self.is_converged:
                    break

```





## XXX

