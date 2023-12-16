# fate组件注册分析

声明`export  FATE_FLOW_PATH=$FATE_PROJECT_BASE/fateflow`
`export  FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE=$FATE_FLOW_PATH/python/fate_flow`

## 服务器启动过程，组件初始化，获得注册组件过程分析


入口`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow_server.py`

```python
    ComponentRegistry.load()
    default_algorithm_provider = ProviderManager.register_default_providers()
    RuntimeConfig.set_component_provider(default_algorithm_provider)
    ComponentRegistry.load()
```

`ComponentRegistry.load()`从t_component_info表中加载组件信息，类似缓存

### ComponentRegistry

来源`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/db/component_registry.py`

```python
class ComponentRegistry:
    REGISTRY = {}

    @classmethod
    def load(cls):
        component_registry = cls.get_from_db(file_utils.load_json_conf_real_time(FATE_FLOW_DEFAULT_COMPONENT_REGISTRY_PATH))
        cls.REGISTRY.update(component_registry)
        for provider_name, provider_info in cls.REGISTRY.get("providers", {}).items():
            if not ComponentProviderName.valid(provider_name):
                raise Exception(f"not support component provider: {provider_name}")
        cls.REGISTRY["providers"] = cls.REGISTRY.get("providers", {})
        cls.REGISTRY["components"] = cls.REGISTRY.get("components", {})
        RuntimeConfig.load_component_registry()

    @classmethod
    def register_provider(cls, provider: ComponentProvider):
        provider_interface = provider_utils.get_provider_interface(provider)
        support_components = provider_interface.get_names()
        components = {}
        for component_alias, info in support_components.items():
            component_name = component_alias.lower()
            if component_name not in components:
                components[component_name] = info
            elif components[component_name].get("module") != info.get("module"):
                raise ValueError(f"component {component_name} have different module info")
            components[component_name]["alias"] = components[component_name].get("alias", set())
            components[component_name]["alias"].add(component_alias)
        register_info = {
            "default": {
                "version": provider.version
            }
        }
        register_info = cls.get_providers().get(provider.name, register_info)
        register_info[provider.version] = {
                "path": provider.path,
                "class_path": provider.class_path,
                "components": components
        }
        cls.REGISTRY["providers"][provider.name] = register_info
        return components

    @classmethod
    def register_components(cls, provider_name, components: dict):
        for component_name, info in components.items():
            if component_name not in cls.REGISTRY["components"]:
                cls.REGISTRY["components"][component_name] = {
                    "default_provider": provider_name,
                    "support_provider": [],
                    "alias": info["alias"]
                }
            if provider_name not in cls.REGISTRY["components"][component_name]["support_provider"]:
                # do not use set because the json format is not supported
                cls.REGISTRY["components"][component_name]["support_provider"].append(provider_name)
                for component_alias in info["alias"]:
                    cls.REGISTRY["components"][component_alias] = cls.REGISTRY["components"][component_name]

    @classmethod
    def dump(cls):
        cls.save_to_db()

    @classmethod
    @DB.connection_context()
    @DB.lock("component_register")
    def save_to_db(cls):
        # save component registry info
        for provider_name, provider_group_info in cls.REGISTRY["providers"].items():
            for version, version_register_info in provider_group_info.items():
                if version != "default":
                    version_info = {
                        "f_path": version_register_info.get("path"),
                        "f_python": version_register_info.get("python", ""),
                        "f_class_path": version_register_info.get("class_path"),
                        "f_version": version,
                        "f_provider_name": provider_name
                    }
                    cls.safe_save(ComponentProviderInfo, version_info, f_version=version, f_provider_name=provider_name)
                    for component_name, component_info in version_register_info.get("components").items():
                        component_registry_info = {
                            "f_version": version,
                            "f_provider_name": provider_name,
                            "f_component_name": component_name,
                            "f_module": component_info.get("module")
                        }
                        cls.safe_save(ComponentRegistryInfo, component_registry_info, f_version=version,
                                        f_provider_name=provider_name, f_component_name=component_name)

        for component_name, info in cls.REGISTRY["components"].items():
            component_info = {
                "f_component_name": component_name,
                "f_default_provider": info.get("default_provider"),
                "f_support_provider": info.get("support_provider"),
                "f_component_alias": info.get("alias"),
            }
            cls.safe_save(ComponentInfo, component_info, f_component_name=component_name)

    @classmethod
    def safe_save(cls, model, defaults, **kwargs):
        entity_model, status = model.get_or_create(
            **kwargs,
            defaults=defaults)
        if status is False:
            for key in defaults:
                setattr(entity_model, key, defaults[key])
            entity_model.save(force_insert=False)

    @classmethod
    @DB.connection_context()
    def get_from_db(cls, component_registry):
        # get component registry info
        # 启动过程被调用两次，第一次什么都没有
        component_list = ComponentInfo.select()
        for component in component_list:
            component_registry["components"][component.f_component_name] = {
                "default_provider": component.f_default_provider,
                "support_provider": component.f_support_provider,
                "alias": component.f_component_alias
            }
            for component_alias in component.f_component_alias:
                component_registry["components"][component_alias] = component_registry["components"][component.f_component_name]

        provider_list = ComponentProviderInfo.select()

        # get key names from `fateflow/conf/component_registry.json`
        default_version_keys = {
            provider_name: default_settings["default_version_key"]
            for provider_name, default_settings in component_registry["default_settings"].items()
            if "default_version_key" in default_settings
        }

        for provider_info in provider_list:
            if provider_info.f_provider_name not in component_registry["providers"]:
                component_registry["providers"][provider_info.f_provider_name] = {
                    "default": {
                        "version": get_versions()[default_version_keys[provider_info.f_provider_name]]
                        if provider_info.f_provider_name in default_version_keys else provider_info.f_version,
                    }
                }

            component_registry["providers"][provider_info.f_provider_name][provider_info.f_version] = {
                "path": provider_info.f_path,
                "python": provider_info.f_python,
                "class_path": provider_info.f_class_path
            }
            modules_list = ComponentRegistryInfo.select().where(
                ComponentRegistryInfo.f_provider_name == provider_info.f_provider_name,
                ComponentRegistryInfo.f_version == provider_info.f_version
            )
            modules = {}
            for module in modules_list:
                modules[module.f_component_name] = {"module": module.f_module}
                for component_alias in component_registry["components"][module.f_component_name]["alias"]:
                    modules[component_alias] = modules[module.f_component_name]
            component_registry["providers"][provider_info.f_provider_name][provider_info.f_version]["components"] = modules
        return component_registry

    @classmethod
    def get_providers(cls):
        return cls.REGISTRY.get("providers", {})

    @classmethod
    def get_components(cls):
        return cls.REGISTRY.get("components", {})

    @classmethod
    def get_provider_components(cls, provider_name, provider_version):
        return cls.get_providers()[provider_name][provider_version]["components"]

    @classmethod
    def get_default_class_path(cls):
        return ComponentRegistry.REGISTRY["default_settings"]["class_path"]
```

#### ComponentRegistry.load()

```python
    @classmethod
    def load(cls):
        component_registry = cls.get_from_db(file_utils.load_json_conf_real_time(FATE_FLOW_DEFAULT_COMPONENT_REGISTRY_PATH))
        cls.REGISTRY.update(component_registry)
        for provider_name, provider_info in cls.REGISTRY.get("providers", {}).items():
            if not ComponentProviderName.valid(provider_name):
                raise Exception(f"not support component provider: {provider_name}")
        cls.REGISTRY["providers"] = cls.REGISTRY.get("providers", {})
        cls.REGISTRY["components"] = cls.REGISTRY.get("components", {})
        RuntimeConfig.load_component_registry()
```

`FATE_FLOW_CONF_PATH = os.path.join(get_fate_flow_directory(), "conf")`

`FATE_FLOW_DEFAULT_COMPONENT_REGISTRY_PATH = os.path.join(FATE_FLOW_CONF_PATH, "component_registry.json")`

可知，`FATE_FLOW_DEFAULT_COMPONENT_REGISTRY_PATH`是`$FATE_FLOW_PATH/conf/component_registry.json`

component_registry.json 
```
{
  "components": {
  },
  "providers": {
  },
  "default_settings": {
    "fate_flow":{
      "default_version_key": "FATEFlow"
    },
    "fate": {
      "default_version_key": "FATE"
    },
    "class_path": {
      "interface": "components.components.Components",
      "feature_instance": "feature.instance.Instance",
      "feature_vector": "feature.sparse_vector.SparseVector",
      "model": "protobuf.generated",
      "model_migrate": "protobuf.model_migrate.model_migrate",
      "homo_model_convert": "protobuf.homo_model_convert.homo_model_convert",
      "anonymous_generator": "util.anonymous_generator_util.Anonymous",
      "data_format": "util.data_format_preprocess.DataFormatPreProcess",
      "hetero_model_merge": "protobuf.model_merge.merge_hetero_models.hetero_model_merge",
      "extract_woe_array_dict": "protobuf.model_migrate.binning_model_migrate.extract_woe_array_dict",
      "merge_woe_array_dict": "protobuf.model_migrate.binning_model_migrate.merge_woe_array_dict"
    }
  }
}
```

`load_json_conf_real_time`就是加载这个json文件并将其转化为`dict`

```python
from enum import IntEnum, Enum


class CustomEnum(Enum):
    @classmethod
    def valid(cls, value):
        try:
            cls(value)
            return True
        except:
            return False

    @classmethod
    def values(cls):
        return [member.value for member in cls.__members__.values()]

    @classmethod
    def names(cls):
        return [member.name for member in cls.__members__.values()]

class ComponentProviderName(CustomEnum):
    FATE = "fate"
    FATE_FLOW = "fate_flow"
    FATE_SQL = "fate_sql"
```



然后会对providers中的provider_name字段做一个命名解析，`ComponentProviderName.valid(provider_name)`会判断`provider_name`是否是枚举常量，也就是说`provider_name`是否是`"fate"`，`"fate_flow"`和`"fate_sql"`中的任意一个。

后续会将读取到的数据更新到`REGISTRY`属性中，也就是将相关的组件信息，包括储存位置加载到了运行时的内存变量中。



#### ComponentRegistry.get_from_db()

``````python
    @classmethod
    @DB.connection_context()
    def get_from_db(cls, component_registry):
        # get component registry info
        # 启动过程被调用两次，第一次什么都没有
        component_list = ComponentInfo.select()
        for component in component_list:    # 遍历t_component_info表中的所有表项
            component_registry["components"][component.f_component_name] = {    # 将每个表项的信息存入component_registry["components"]中
                "default_provider": component.f_default_provider,
                "support_provider": component.f_support_provider,
                "alias": component.f_component_alias
            }
            for component_alias in component.f_component_alias:     # 遍历每个表项的别名
                # 将别名也存入component_registry["components"]中，并且指向原项
                component_registry["components"][component_alias] = component_registry["components"][component.f_component_name]

        provider_list = ComponentProviderInfo.select()

        # get key names from `fateflow/conf/component_registry.json`
        default_version_keys = {
            provider_name: default_settings["default_version_key"]
            for provider_name, default_settings in component_registry["default_settings"].items()
            if "default_version_key" in default_settings
        }

        # default_version_keys = {
        #     "fate": "FATE",
        #     "fate_flow": "FATEFlow"
        # }
        # 两个的模块路径都是$FATE_PROJECT_BASE/python/federatedml

        for provider_info in provider_list:    # 遍历t_component_provider_info表中的所有表项，默认情况下只有两个表项，fate和fate_flow
            if provider_info.f_provider_name not in component_registry["providers"]:    # 如果provider_info.f_provider_name不在component_registry["providers"]中
                # 给每个provider增加一个default字段
                component_registry["providers"][provider_info.f_provider_name] = {
                    "default": {
                        "version": get_versions()[default_version_keys[provider_info.f_provider_name]]
                        if provider_info.f_provider_name in default_version_keys else provider_info.f_version,
                    }
                }

            # 给每个表项对应的provider的对应版本记录相关信息，也就是从数据库中读取的信息
            component_registry["providers"][provider_info.f_provider_name][provider_info.f_version] = {
                "path": provider_info.f_path,   # 提供者模块的路径
                "python": provider_info.f_python,   # 提供者模块的python版本？目前见到的赋值都是空字符串
                "class_path": provider_info.f_class_path    # 提供者模块的类路径
            }
            modules_list = ComponentRegistryInfo.select().where(
                ComponentRegistryInfo.f_provider_name == provider_info.f_provider_name,
                ComponentRegistryInfo.f_version == provider_info.f_version
            )   # 从t_component_registry_info表中读取f_provider_name和f_version对应的所有表项
            modules = {}
            for module in modules_list:    # 遍历所有表项
                modules[module.f_component_name] = {"module": module.f_module}  # 将每个表项的信息存入modules中
                for component_alias in component_registry["components"][module.f_component_name]["alias"]:  # 遍历每个表项的别名
                    modules[component_alias] = modules[module.f_component_name]  # 将别名也存入modules中，并且指向原项
            component_registry["providers"][provider_info.f_provider_name][provider_info.f_version]["components"] = modules # 每个provider的对应版本记录相关信息
        return component_registry


``````

load()函数会被调用两次，如果是初次启动fate flow server，那么数据库是空的，ComponentRegistry的REGISTRY属性在第一次load的时候和component_registry.json中的内容一样。

```python
{'components': {},
 'default_settings': {'class_path': {'anonymous_generator': 'util.anonymous_generator_util.Anonymous',
                                     'data_format': 'util.data_format_preprocess.DataFormatPreProcess',
                                     'extract_woe_array_dict': 'protobuf.model_migrate.binning_model_migrate.extract_woe_array_dict',
                                     'feature_instance': 'feature.instance.Instance',
                                     'feature_vector': 'feature.sparse_vector.SparseVector',
                                     'hetero_model_merge': 'protobuf.model_merge.merge_hetero_models.hetero_model_merge',
                                     'homo_model_convert': 'protobuf.homo_model_convert.homo_model_convert',
                                     'interface': 'components.components.Components',
                                     'merge_woe_array_dict': 'protobuf.model_migrate.binning_model_migrate.merge_woe_array_dict',
                                     'model': 'protobuf.generated',
                                     'model_migrate': 'protobuf.model_migrate.model_migrate'},
                      'fate': {'default_version_key': 'FATE'},
                      'fate_flow': {'default_version_key': 'FATEFlow'}},
 'providers': {}}

```

### ProviderManager

分析`default_algorithm_provider = ProviderManager.register_default_providers()`的调用过程

`ProviderManager`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/manager/provider_manager.py`



#### ProviderManager.register_default_providers()

```python
    @classmethod
    def register_default_providers(cls):
        code, result = cls.register_fate_flow_provider()
        if code != 0:
            raise Exception(f"register fate flow tools component failed")
        code, result, provider = cls.register_default_fate_provider()
        if code != 0:
            raise Exception(f"register default fate algorithm component failed")
        return provider
```

#### ProviderManager.register_fate_flow_provider()

```python
    @classmethod
    def register_fate_flow_provider(cls):
        # 注册fate_flow的provider，在注册的时候，会把fate_flow的组件相关信息写入到t_component_registry和t_component_provider_info表中
        provider = cls.get_fate_flow_provider() # 获得fate_flow的provider
        return WorkerManager.start_general_worker(worker_name=WorkerName.PROVIDER_REGISTRAR, provider=provider, run_in_subprocess=False)

```

#### ProviderManager.get_fate_flow_provider()

```python
    @classmethod
    def get_fate_flow_provider(cls):
        path = get_fate_flow_python_directory("fate_flow")  # fate_flow\python\fate_flow    
        provider = ComponentProvider(name="fate_flow", version=get_versions()["FATEFlow"], path=path, class_path=ComponentRegistry.get_default_class_path())
        return provider
```

#### ComponentProvider

`ComponentProvider`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/entity/_component_provider.py`

```python
class ComponentProvider(BaseEntity):
    def __init__(self, name: str, version: str, path: str, class_path: dict, _python_env="",**kwargs):
        if not ComponentProviderName.valid(name):
            raise ValueError(f"not support {name} provider")
        self._name = name
        self._version = version
        self._path = os.path.abspath(path)
        self._class_path = class_path
        self._python_env = _python_env
        self._env = {}
        self.init_env()

    def init_env(self):
        self._env["PYTHONPATH"] = os.path.dirname(self._path)
        self._env["PYTHON_ENV"] = self.python_env

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def path(self):
        return self._path

    @property
    def class_path(self):
        return self._class_path

    @property
    def env(self):
        return self._env

    @property
    def python_env(self):
        return self._python_env

    def __eq__(self, other):
        return self.name == other.name and self.version == other.version

```

`ComponentProvider`的作用是将组件的信息封装成一个对象，包括组件的名称，版本，路径，类路径，python环境等信息。

##### BaseEntity

`BaseEntity`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/entity/_base.py`

```python
class BaseEntity(BaseType):
    pass

```

##### BaseType

`BaseType`来自`$FATE_PROJECT_BASE/python/fate_arch/common/_types.py`

```python
class BaseType:
    def to_dict(self):
        return dict([(k.lstrip("_"), v) for k, v in self.__dict__.items()])

    def to_dict_with_type(self):
        def _dict(obj):
            module = None
            if issubclass(obj.__class__, BaseType):
                data = {}
                for attr, v in obj.__dict__.items():
                    k = attr.lstrip("_")
                    data[k] = _dict(v)
                module = obj.__module__
            elif isinstance(obj, (list, tuple)):
                data = []
                for i, vv in enumerate(obj):
                    data.append(_dict(vv))
            elif isinstance(obj, dict):
                data = {}
                for _k, vv in obj.items():
                    data[_k] = _dict(vv)
            else:
                data = obj
            return {"type": obj.__class__.__name__, "data": data, "module": module}
        return _dict(self)
```

`BaseType`实现的就是将对象属性转化成字典键值对，包括任何下划线开头"_"的属性

### WorkerManager

`WorkerManager`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/manager/worker_manager.py`

```python
class WorkerManager:
    ...
```

#### WorkerManager.start_general_worker(...)

```python
    @classmethod
    def start_general_worker(cls, worker_name: WorkerName, job_id="", role="", party_id=0, provider: ComponentProvider = None,
                             initialized_config: dict = None, run_in_subprocess=True, **kwargs):
        if RuntimeConfig.DEBUG:
            run_in_subprocess = True
        participate = locals()
        worker_id, config_dir, log_dir = cls.get_process_dirs(worker_name=worker_name,
                                                              job_id=job_id,
                                                              role=role,
                                                              party_id=party_id)
        if worker_name in [WorkerName.PROVIDER_REGISTRAR, WorkerName.DEPENDENCE_UPLOAD]:
            if not provider:
                raise ValueError("no provider argument")
            config = {
                "provider": provider.to_dict()
            }
            if worker_name == WorkerName.PROVIDER_REGISTRAR:
                from fate_flow.worker.provider_registrar import ProviderRegistrar
                module = ProviderRegistrar
                module_file_path = sys.modules[ProviderRegistrar.__module__].__file__
                specific_cmd = []
            elif worker_name == WorkerName.DEPENDENCE_UPLOAD:
                from fate_flow.worker.dependence_upload import DependenceUpload
                module = DependenceUpload
                module_file_path = sys.modules[DependenceUpload.__module__].__file__
                specific_cmd = [
                    '--dependence_type', kwargs.get("dependence_type")
                ]
            provider_info = provider.to_dict()
        elif worker_name is WorkerName.TASK_INITIALIZER:
            if not initialized_config:
                raise ValueError("no initialized_config argument")
            config = initialized_config
            from fate_flow.worker.task_initializer import TaskInitializer
            module = TaskInitializer
            module_file_path = sys.modules[TaskInitializer.__module__].__file__
            specific_cmd = []
            provider_info = initialized_config["provider"]
        else:
            raise Exception(f"not support {worker_name} worker")
        config_path, result_path = cls.get_config(config_dir=config_dir, config=config, log_dir=log_dir)

        process_cmd = [
            sys.executable or "python3",
            module_file_path,
            "--config", config_path,
            '--result', result_path,
            "--log_dir", log_dir,
            "--parent_log_dir", os.path.dirname(log_dir),
            "--worker_id", worker_id,
            "--run_ip", RuntimeConfig.JOB_SERVER_HOST,
            "--job_server", f"{RuntimeConfig.JOB_SERVER_HOST}:{RuntimeConfig.HTTP_PORT}",
        ]

        if job_id:
            process_cmd.extend([
                "--job_id", job_id,
                "--role", role,
                "--party_id", party_id,
            ])

        process_cmd.extend(specific_cmd)
        if run_in_subprocess:
            p = process_utils.run_subprocess(job_id=job_id, config_dir=config_dir, process_cmd=process_cmd,
                                             added_env=cls.get_env(job_id, provider_info), log_dir=log_dir,
                                             cwd_dir=config_dir, process_name=worker_name.value, process_id=worker_id)
            participate["pid"] = p.pid
            if job_id and role and party_id:
                logger = schedule_logger(job_id)
                msg = f"{worker_name} worker {worker_id} subprocess {p.pid}"
            else:
                logger = stat_logger
                msg = f"{worker_name} worker {worker_id} subprocess {p.pid}"
            logger.info(ready_log(msg=msg, role=role, party_id=party_id))

            # asynchronous
            if worker_name in [WorkerName.DEPENDENCE_UPLOAD]:
                if kwargs.get("callback") and kwargs.get("callback_param"):
                    callback_param = {}
                    participate.update(participate.get("kwargs", {}))
                    for k, v in participate.items():
                        if k in kwargs.get("callback_param"):
                            callback_param[k] = v
                    kwargs.get("callback")(**callback_param)
            else:
                try:
                    p.wait(timeout=120)
                    if p.returncode == 0:
                        logger.info(successful_log(msg=msg, role=role, party_id=party_id))
                    else:
                        logger.info(failed_log(msg=msg, role=role, party_id=party_id))
                    if p.returncode == 0:
                        return p.returncode, load_json_conf(result_path)
                    else:
                        std_path = process_utils.get_std_path(log_dir=log_dir, process_name=worker_name.value, process_id=worker_id)
                        raise Exception(f"run error, please check logs: {std_path}, {log_dir}/INFO.log")
                except subprocess.TimeoutExpired as e:
                    err = failed_log(msg=f"{msg} run timeout", role=role, party_id=party_id)
                    logger.exception(err)
                    raise Exception(err)
                finally:
                    try:
                        p.kill()
                        p.poll()
                    except Exception as e:
                        logger.exception(e)
        else:
            kwargs = cls.cmd_to_func_kwargs(process_cmd)
            code, message, result = module().run(**kwargs)
            if code == 0:
                return code, result
            else:
                raise Exception(message)

```

`$FATE_PROJECT_BASE/fate_flow/{worker_name.value}/{worker_id}/config`记录了provider的属性

```bash
(fate_venv) lab@lab-virtual-machine:~/federated_learning/fate/from_src_build/FATE/fateflow$ cat $FATE_PROJECT_BASE/fateflow/provider_registrar/1bcbbe523f1511ee87cde1d2a5b270dd/config.json 
{"provider": {"name": "fate_flow", "version": "1.11.1", "path": "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow", "class_path": {"interface": "components.components.Components", "feature_instance": "feature.instance.Instance", "feature_vector": "feature.sparse_vector.SparseVector", "model": "protobuf.generated", "model_migrate": "protobuf.model_migrate.model_migrate", "homo_model_convert": "protobuf.homo_model_convert.homo_model_convert", "anonymous_generator": "util.anonymous_generator_util.Anonymous", "data_format": "util.data_format_preprocess.DataFormatPreProcess", "hetero_model_merge": "protobuf.model_merge.merge_hetero_models.hetero_model_merge", "extract_woe_array_dict": "protobuf.model_migrate.binning_model_migrate.extract_woe_array_dict", "merge_woe_array_dict": "protobuf.model_migrate.binning_model_migrate.merge_woe_array_dict"}, "python_env": "", "env": {"PYTHONPATH": "/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python", "PYTHON_ENV": ""}}}
```

<hr>

```python
        if run_in_subprocess:
            ...
        else:
            kwargs = cls.cmd_to_func_kwargs(process_cmd)
            code, message, result = module().run(**kwargs)
            if code == 0:
                return code, result
            else:
                raise Exception(message)

```



动态调试后的`process_cmd`和`kwargs`

```python
process_cmd = ['/home/lab/federated_learning/fate/from_src_build/FATE/fate_venv/bin/python',
'/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow/worker/provider_registrar.py',
 '--config',
'/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/provider_registrar/713f38d03f1711ee87cde1d2a5b270dd/config.json',
 '--result',
'/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/provider_registrar/713f38d03f1711ee87cde1d2a5b270dd/result.json',
 '--log_dir',
'/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/logs/provider_registrar/713f38d03f1711ee87cde1d2a5b270dd',
 '--parent_log_dir',
 '/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/logs/provider_registrar',
 '--worker_id',
 '713f38d03f1711ee87cde1d2a5b270dd',
 '--run_ip',
 '127.0.0.1',
 '--job_server',
 '127.0.0.1:9380']
```
```python
kwargs = {'config': '/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/provider_registrar/713f38d03f1711ee87cde1d2a5b270dd/config.json',
 'result': '/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/provider_registrar/713f38d03f1711ee87cde1d2a5b270dd/result.json',
 'log_dir': '/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/logs/provider_registrar/713f38d03f1711ee87cde1d2a5b270dd',
 'parent_log_dir': '/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/logs/provider_registrar',
 'worker_id': '713f38d03f1711ee87cde1d2a5b270dd',
 'run_ip': '127.0.0.1',
 'job_server': '127.0.0.1:9380'}
```

`code, message, result = module().run(**kwargs)`	相当于

`code, message, result = ProviderRegistrar().run(**kwargs)` 

### ProviderRegistrar

`ProviderRegistrar`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/worker/provider_registrar.py`

```python
class ProviderRegistrar(BaseWorker):
	...

```



##### BaseWorker

`BaseWorker`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/worker/base_worker.py`

```python
class BaseWorker:
    def __init__(self):
        self.args: WorkerArgs = None
        self.run_pid = None
        self.report_info = {}

    def run(self, **kwargs):
        result = {}
        code = 0
        message = ""
        start_time = current_timestamp()
        self.run_pid = os.getpid()
        try:
            self.args = self.get_args(**kwargs)
            if self.args.model_path:
                os.environ["MODEL_PATH"] = self.args.model_path
            RuntimeConfig.init_env()
            role = ProcessRole(os.getenv("PROCESS_ROLE"))
            append_to_parent_log = True
            if self.args.is_deepspeed:
                role = ProcessRole(ProcessRole.WORKER.value)
                append_to_parent_log = False
            RuntimeConfig.set_process_role(role)
            if RuntimeConfig.PROCESS_ROLE == ProcessRole.WORKER:
                LoggerFactory.LEVEL = logging.getLevelName(os.getenv("FATE_LOG_LEVEL", "INFO"))
                if os.getenv("EGGROLL_CONTAINER_LOGS_DIR"):
                    # eggroll deepspeed
                    self.args.parent_log_dir = os.path.dirname(os.getenv("EGGROLL_CONTAINER_LOGS_DIR"))
                    self.args.log_dir = os.getenv("EGGROLL_CONTAINER_LOGS_DIR")
                LoggerFactory.set_directory(directory=self.args.log_dir, parent_log_dir=self.args.parent_log_dir,
                                            append_to_parent_log=append_to_parent_log, force=True)
                LOGGER.info(f"enter {self.__class__.__name__} worker in subprocess, pid: {self.run_pid}")
            else:
                LOGGER.info(f"enter {self.__class__.__name__} worker in driver process, pid: {self.run_pid}")
            LOGGER.info(f"log level: {logging.getLevelName(LoggerFactory.LEVEL)}")
            for env in {"VIRTUAL_ENV", "PYTHONPATH", "SPARK_HOME", "FATE_DEPLOY_BASE", "PROCESS_ROLE", "FATE_JOB_ID"}:
                LOGGER.info(f"{env}: {os.getenv(env)}")
            if self.args.job_server:
                RuntimeConfig.init_config(JOB_SERVER_HOST=self.args.job_server.split(':')[0],
                                          HTTP_PORT=self.args.job_server.split(':')[1])
            if not RuntimeConfig.LOAD_COMPONENT_REGISTRY:
                ComponentRegistry.load()
            if not RuntimeConfig.LOAD_CONFIG_MANAGER:
                ConfigManager.load()
            result = self._run()
        except Exception as e:
            LOGGER.exception(e)
            traceback.print_exc()
            try:
                self._handle_exception()
            except Exception as e:
                LOGGER.exception(e)
            code = 1
            message = exception_to_trace_string(e)
        finally:
            if self.args and self.args.result:
                if not self.args.is_deepspeed:
                    dump_json_conf(result, self.args.result)
            end_time = current_timestamp()
            LOGGER.info(f"worker {self.__class__.__name__}, process role: {RuntimeConfig.PROCESS_ROLE}, pid: {self.run_pid}, elapsed: {end_time - start_time} ms")
            if RuntimeConfig.PROCESS_ROLE == ProcessRole.WORKER:
                sys.exit(code)
            if self.args and self.args.is_deepspeed:
                sys.exit(code)
            else:
                return code, message, result

    def _run(self):
        raise NotImplementedError

    def _handle_exception(self):
        pass

    @staticmethod
    def get_args(**kwargs):
        if kwargs:
            return WorkerArgs(**kwargs)
        else:
            parser = argparse.ArgumentParser()
            for arg in WorkerArgs().to_dict():
                parser.add_argument(f"--{arg}", required=False)
            return WorkerArgs(**parser.parse_args().__dict__)


```

run()初始会进行一些基本的配置，主要的逻辑还是子类覆盖的方法`ProviderRegistrar._run()`中

##### ProcessRole

```python
class ProcessRole(CustomEnum):
    DRIVER = "driver"
    WORKER = "worker"
```

#### ProviderRegistrar._run()

```python
    def _run(self):
        provider = ComponentProvider(**self.args.config.get("provider"))    # 获取provider配置
        support_components = ComponentRegistry.register_provider(provider)  # 注册provider，在这个过程中，会导入相关的模块！
        ComponentRegistry.register_components(provider.name, support_components)    # 通过以上信息，注册组件，将组件信息更新到ComponentRegistry.REGISTRY中
        ComponentRegistry.dump()    # 将组件信息写入到数据库中
        stat_logger.info(json_dumps(ComponentRegistry.REGISTRY, indent=4))
```

读取`$FATE_PROJECT_BASE/fateflow/provider_registrar/713f38d03f1711ee87cde1d2a5b270dd/config.json`中的provider属性，并作为参数传入`ComponentProvider`的构造函数，创建一个新的`ComponentProvider`对象。


`self.args.config.get("provider")`

```json
{'class_path': {'anonymous_generator': 'util.anonymous_generator_util.Anonymous',
                'data_format': 'util.data_format_preprocess.DataFormatPreProcess',
                'extract_woe_array_dict': 'protobuf.model_migrate.binning_model_migrate.extract_woe_array_dict',
                'feature_instance': 'feature.instance.Instance',
                'feature_vector': 'feature.sparse_vector.SparseVector',
                'hetero_model_merge': 'protobuf.model_merge.merge_hetero_models.hetero_model_merge',
                'homo_model_convert': 'protobuf.homo_model_convert.homo_model_convert',
                'interface': 'components.components.Components',
                'merge_woe_array_dict': 'protobuf.model_migrate.binning_model_migrate.merge_woe_array_dict',
                'model': 'protobuf.generated',
                'model_migrate': 'protobuf.model_migrate.model_migrate'},
 'env': {'PYTHONPATH': '/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python',
         'PYTHON_ENV': ''},
 'name': 'fate_flow',
 'path': '/home/lab/federated_learning/fate/from_src_build/FATE/fateflow/python/fate_flow',
 'python_env': '',
 'version': '1.11.1'}

```

然后通过`ComponentRegistry.register_provider(provider)`进行组件注册

#### ComponentRegistry.register_provider(...)

```python
    @classmethod
    def register_provider(cls, provider: ComponentProvider):
        # 通过provider注册组件
        provider_interface = provider_utils.get_provider_interface(provider)    # 导入provider的接口 'components.components.Components'
        # provider_interface就是components.components.Components，是一个类
        support_components = provider_interface.get_names()   # 获取provider支持的组件以及对应的模块，返回一个字典，key是组件名，value是对应的模块，在这个过程就会导入所有模块一次，但是无法直接使用。
        components = {}
        for component_alias, info in support_components.items():    # 遍历组件
            component_name = component_alias.lower()    # 组件名转小写
            if component_name not in components:    # 如果组件名不在components中
                components[component_name] = info   # 将组件名和对应的模块添加到components中
            elif components[component_name].get("module") != info.get("module"):    # 如果组件名在components中，但是对应的模块不一样
                raise ValueError(f"component {component_name} have different module info")  # 抛出异常
            components[component_name]["alias"] = components[component_name].get("alias", set())        # 获取组件名对应的别名
            components[component_name]["alias"].add(component_alias)    # 将组件名对应的别名添加到components中
        register_info = {   # 默认的注册信息
            "default": {
                "version": provider.version
            }
        }
        register_info = cls.get_providers().get(provider.name, register_info)       # 获取provider的注册信息
        register_info[provider.version] = {   # 将provider的版本信息添加到register_info中
                "path": provider.path,
                "class_path": provider.class_path,
                "components": components
        }
        cls.REGISTRY["providers"][provider.name] = register_info    # 将register_info添加到cls.REGISTRY["providers"][provider.name]中
        return components
```

##### get_provider_interface(...)

`get_provider_interface`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/component_env_utils/provider_utils.py`


```python
def get_provider_interface(provider: ComponentProvider):
    obj = get_provider_class_object(provider, "interface")
    for i in ('name', 'version', 'path'):
        setattr(obj, f'provider_{i}', getattr(provider, i))
    return obj

def get_provider_class_object(provider: ComponentProvider, class_name, module=False):
    class_path = get_provider_class_import_path(provider, class_name)
    if module:
        return importlib.import_module(".".join(class_path))
    else:
        return getattr(importlib.import_module(".".join(class_path[:-1])), class_path[-1])


def get_provider_class_import_path(provider: ComponentProvider, class_name):
    return f"{pathlib.Path(provider.path).name}.{provider.class_path.get(class_name)}".split(".")

```

最后返回的是一个`components.components.Components`对象

#### components.py

`components.py`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow/components/components.py`


```python

def _get_module_name_by_path(path, base):
    return '.'.join(path.resolve().relative_to(base.resolve()).with_suffix('').parts)


def _search_components(path, base):
    try:
        module_name = _get_module_name_by_path(path, base)
        module = importlib.import_module(module_name)
    except ImportError as e:
        # or skip ?
        raise e
    _obj_pairs = inspect.getmembers(module, lambda obj: isinstance(obj, ComponentMeta))
    return _obj_pairs, module_name


class Components:
    provider_version = None
    provider_name = None
    provider_path = None

    @classmethod
    def _module_base(cls):
        return Path(cls.provider_path).resolve().parent

    @classmethod
    def _components_base(cls):
        return Path(cls.provider_path, 'components').resolve()

    @classmethod
    def get_names(cls) -> typing.Dict[str, dict]:
        # 在这里查找所有支持的组件，虽然有importlib.import_module，但是只用于查找，导入的obj在这里暂时不会被使用，但也是不可或缺的一步，因为这样才能确保所有的组件都被注册到ComponentMeta中（通过运行xxx_cpn_meta = ComponentMeta("xxx")），所以要提前导入所有组件，但是只有导入的obj的信息被注册到ComponentMeta中，然后在get的时候再导入真正需要的组件。
        names = {}
        for p in cls._components_base().glob("**/*.py"):
            obj_pairs, module_name = _search_components(p.resolve(), cls._module_base())
            for name, obj in obj_pairs:
                names[obj.name] = {"module": module_name}
                LOGGER.info(f"component register {obj.name} with cache info {module_name}")
        # 断点
        return names

    @classmethod
    def get(cls, name: str, cache) -> ComponentMeta:
        if cache:
            importlib.import_module(cache[name]["module"])
        else:
            for p in cls._components_base().glob("**/*.py"):
                module_name = _get_module_name_by_path(p, cls._module_base())
                importlib.import_module(module_name)

        cpn = ComponentMeta.get_meta(name)
        return cpn

```

在`get_names`中会查找所有支持的组件，虽然有`importlib.import_module`，但是只用于查找，导入的obj在这里暂时不会被使用，但也是不可或缺的一步，因为这样才能确保所有的组件都被注册到`ComponentMeta`中（通过运行`xxx_cpn_meta = ComponentMeta("xxx")`），所以要提前导入所有组件，但是只有导入的obj的信息被注册到`ComponentMeta`中，然后在get的时候再导入真正需要的组件。

`get_names`return之前打下断点，获取到的`ComponentMeta._ComponentMeta__name_to_obj`

```python
{'Download': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f377f0>, 'Reader': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f401f0>, 'ModelLoader': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f379d0>, 'ApiReader': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f40b20>, 'ModelStore': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f03430>, 'ModelRestore': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f034c0>, 'Upload': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f0b6a0>, 'CacheLoader': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f0bf10>, 'Writer': <fate_flow.components._base.ComponentMeta object at 0x7f69d9f0bc40>}
```

可以发现实际上是由很多个`ComponentMeta`对象，对应不同的组件类来实现对象到类的绑定（映射）。

实现了动态导入绑定，妙哉。

#### ComponentMeta

```python

class ComponentMeta:
    __name_to_obj: typing.Dict[str, "ComponentMeta"] = {}

    def __init__(self, name) -> None:
        self.name = name
        self._role_to_runner_cls = {}
        self._param_cls = None

        self.__name_to_obj[name] = self	# 每个组件文件都有xxx_cpn_meta = ComponentMeta("xxx")，以此来实现动态绑定

    @property
    def bind_runner(self):
        return _RunnerDecorator(self)

    @property
    def bind_param(self):
        def _wrap(cls):
            self._param_cls = cls
            return cls

        return _wrap

    def register_info(self):
        return {
            self.name: dict(
                module=self.__module__,
            )
        }

    @classmethod
    def get_meta(cls, name):
        return cls.__name_to_obj[name]

    def _get_runner(self, role: str):
        if role not in self._role_to_runner_cls:
            raise ModuleNotFoundError(
                f"Runner for component `{self.name}` at role `{role}` not found"
            )
        return self._role_to_runner_cls[role]

    def get_run_obj(self, role: str):
        return self._get_runner(role)()

    def get_run_obj_name(self, role: str) -> str:
        return self._get_runner(role).__name__

    def get_param_obj(self, cpn_name: str):
        if self._param_cls is None:
            raise ModuleNotFoundError(f"Param for component `{self.name}` not found")
        param_obj = self._param_cls().set_name(f"{self.name}#{cpn_name}")
        return param_obj

    def get_supported_roles(self):
        roles = set(self._role_to_runner_cls.keys())
        if not roles:
            raise ModuleNotFoundError(f"roles for {self.name} is empty")
        return roles


```

<hr>
fate_flow provider的组件注册过程到此结束。
fate provides的组件注册过程类似，除开文件路径不同，差异不大。
<hr>


## 至此，fate框架的组件注册过程已经分析完毕。




