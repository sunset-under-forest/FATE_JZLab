# fate flow server 配置信息

声明`export  FATE_FLOW_PATH=$FATE_PROJECT_BASE/fateflow`
`export  FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE=$FATE_FLOW_PATH/python/fate_flow`

## 配置信息初始化

入口`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/fate_flow_server.py`


### 在初始化数据库相关操作后，会进行运行时配置信息的设置。

```python
    # init runtime config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default=False, help="fate flow version", action='store_true')
    parser.add_argument('--debug', default=False, help="debug mode", action='store_true')
    args = parser.parse_args()
    if args.version:
        print(get_versions())
        sys.exit(0)
    # todo: add a general init steps?
    RuntimeConfig.DEBUG = args.debug
    if RuntimeConfig.DEBUG:
        stat_logger.info("run on debug mode")
    ConfigManager.load()
    RuntimeConfig.init_env()
    RuntimeConfig.init_config(JOB_SERVER_HOST=HOST, HTTP_PORT=HTTP_PORT)
    RuntimeConfig.set_process_role(ProcessRole.DRIVER)

    RuntimeConfig.set_service_db(service_db())
    RuntimeConfig.SERVICE_DB.register_flow()
    RuntimeConfig.SERVICE_DB.register_models()

```

RuntimeConfig来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/db/runtime_config.py`，是一个静态类，用于存储运行时的配置信息。

```python
class RuntimeConfig(ReloadConfigBase):
    DEBUG = None
    WORK_MODE = None
    COMPUTING_ENGINE = None
    FEDERATION_ENGINE = None
    FEDERATED_MODE = None

    JOB_QUEUE = None
    USE_LOCAL_DATABASE = False
    HTTP_PORT = None
    JOB_SERVER_HOST = None
    JOB_SERVER_VIP = None
    IS_SERVER = False
    PROCESS_ROLE = None
    ENV = dict()
    COMPONENT_PROVIDER: ComponentProvider = None
    SERVICE_DB = None
    LOAD_COMPONENT_REGISTRY = False
    LOAD_CONFIG_MANAGER = False

    @classmethod
    def init_config(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)

    @classmethod
    def init_env(cls):
        cls.ENV.update(get_versions())

    @classmethod
    def load_component_registry(cls):
        cls.LOAD_COMPONENT_REGISTRY = True

    @classmethod
    def load_config_manager(cls):
        cls.LOAD_CONFIG_MANAGER = True

    @classmethod
    def get_env(cls, key):
        return cls.ENV.get(key, None)

    @classmethod
    def get_all_env(cls):
        return cls.ENV

    @classmethod
    def set_process_role(cls, process_role: ProcessRole):
        cls.PROCESS_ROLE = process_role

    @classmethod
    def set_component_provider(cls, component_provider: ComponentProvider):
        cls.COMPONENT_PROVIDER = component_provider

    @classmethod
    def set_service_db(cls, service_db):
        cls.SERVICE_DB = service_db

```

它是ReloadConfigBase的子类，ReloadConfigBase属于`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/db/reload_config_base.py`，有两个类方法，get_all和get，分别用于获取所有配置信息和获取指定配置信息，类似于字典的操作。


```python
class ReloadConfigBase:
    @classmethod
    def get_all(cls):
        configs = {}
        for k, v in cls.__dict__.items():
            if not callable(getattr(cls, k)) and not k.startswith("__") and not k.startswith("_"):
                configs[k] = v
        return configs

    @classmethod
    def get(cls, config_name):
        return getattr(cls, config_name) if hasattr(cls, config_name) else None
```

`ReloadConfigBase`的子类除了`RuntimeConfig`还有`JobDefaultConfig`，`ServerRegistry`和`ServiceRegistry`。

### 随后会调用ConfigManager.load()方法，加载其他的配置信息。

ConfigManager来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/db/config_manager.py`，只有一个类方法load，用于加载配置信息。

```python
class ConfigManager:
    @classmethod
    def load(cls):
        configs = {
            "job_default_config": JobDefaultConfig.load(),
            "server_registry": ServerRegistry.load(),
        }
        ResourceManager.initialize()
        RuntimeConfig.load_config_manager()
        return configs

```

#### 首先加载JobDefaultConfig类，来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/db/job_default_config.py`

```python
class JobDefaultConfig(ReloadConfigBase):
    # component provider
    default_component_provider_path = None

    # Resource
    total_cores_overweight_percent = None
    total_memory_overweight_percent = None
    task_parallelism = None
    task_cores = None
    task_memory = None
    max_cores_percent_per_job = None

    # scheduling
    remote_request_timeout = None
    federated_command_trys = None
    job_timeout = None
    end_status_job_scheduling_time_limit = None
    end_status_job_scheduling_updates = None
    auto_retries = None
    auto_retry_delay = None
    federated_status_collect_type = None
    detect_connect_max_retry_count = None
    detect_connect_long_retry_count = None

    # upload
    upload_block_max_bytes = None  # bytes

    # component output
    output_data_summary_count_limit = None

    task_world_size = None
    resource_waiting_timeout = None
    task_process_classpath = None

    @classmethod
    def load(cls):
        conf = file_utils.load_yaml_conf(FATE_FLOW_JOB_DEFAULT_CONFIG_PATH)
        if not isinstance(conf, dict):
            raise ValueError("invalid config file")

        for k, v in conf.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
            else:
                stat_logger.warning(f"job default config not supported {k}")

        return cls.get_all()

```

这里的FATE_FLOW_JOB_DEFAULT_CONFIG_PATH是`$FATE_FLOW_PATH/conf/job_default_config.yaml`，内容如下：

```yaml
# component provider, relative path to get_fate_python_directory
default_component_provider_path: federatedml

# resource
total_cores_overweight_percent: 1  # 1 means no overweight
total_memory_overweight_percent: 1  # 1 means no overweight
task_parallelism: 1
task_cores: 4
task_memory: 0  # mb
max_cores_percent_per_job: 1  # 1 means total

# scheduling
job_timeout: 259200 # s
remote_request_timeout: 30000  # ms
federated_command_trys: 3
end_status_job_scheduling_time_limit: 300000 # ms
end_status_job_scheduling_updates: 1
auto_retries: 0
auto_retry_delay: 1  #seconds
# It can also be specified in the job configuration using the federated_status_collect_type parameter
federated_status_collect_type: PUSH
detect_connect_max_retry_count: 3
detect_connect_long_retry_count: 2

task_process_classpath: true

# upload
upload_block_max_bytes: 104857600 # bytes

#component output
output_data_summary_count_limit: 100

# gpu
task_world_size: 2
```


#### 然后加载ServerRegistry类，来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/db/service_registry.py`


```python
class ServerRegistry(ReloadConfigBase):
    FATEBOARD = None
    FATE_ON_STANDALONE = None
    FATE_ON_EGGROLL = None
    FATE_ON_SPARK = None
    MODEL_STORE_ADDRESS = None
    SERVINGS = None
    FATEMANAGER = None
    STUDIO = None

    @classmethod
    def load(cls):
        cls.load_server_info_from_conf()
        cls.load_server_info_from_db()

    @classmethod
    def load_server_info_from_conf(cls):
        path = Path(file_utils.get_project_base_directory()) / 'conf' / SERVICE_CONF
        conf = file_utils.load_yaml_conf(path)
        if not isinstance(conf, dict):
            raise ValueError('invalid config file')

        local_path = path.with_name(f'local.{SERVICE_CONF}')
        if local_path.exists():
            local_conf = file_utils.load_yaml_conf(local_path)
            if not isinstance(local_conf, dict):
                raise ValueError('invalid local config file')
            conf.update(local_conf)
        for k, v in conf.items():
            if isinstance(v, dict):
                setattr(cls, k.upper(), v)

    @classmethod
    def register(cls, server_name, server_info):
        cls.save_server_info_to_db(server_name, server_info.get("host"), server_info.get("port"), protocol=server_info.get("protocol", "http"))
        setattr(cls, server_name, server_info)

    @classmethod
    def save(cls, service_config):
        update_server = {}
        for server_name, server_info in service_config.items():
            cls.parameter_check(server_info)
            api_info = server_info.pop("api", {})
            for service_name, info in api_info.items():
                ServiceRegistry.save_service_info(
                    server_name, service_name, uri=info.get('uri'),
                    method=info.get('method', 'POST'),
                    server_info=server_info,
                    data=info.get("data", {}),
                    headers=info.get("headers", {}),
                    params=info.get("params", {})
                )
            cls.save_server_info_to_db(server_name, server_info.get("host"), server_info.get("port"), protocol="http")
            setattr(cls, server_name.upper(), server_info)
        return update_server

    @classmethod
    def parameter_check(cls, service_info):
        if "host" in service_info and "port" in service_info:
            cls.connection_test(service_info.get("host"), service_info.get("port"))

    @classmethod
    def connection_test(cls, ip, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        if result != 0:
            raise ConnectionRefusedError(f"connection refused: host {ip}, port {port}")

    @classmethod
    def query(cls, service_name, default=None):
        service_info = getattr(cls, service_name, default)
        if not service_info:
            service_info = conf_utils.get_base_config(service_name, default)
        return service_info

    @classmethod
    @DB.connection_context()
    def query_server_info_from_db(cls, server_name=None) -> [ServerRegistryInfo]:
        if server_name:
            server_list = ServerRegistryInfo.select().where(ServerRegistryInfo.f_server_name==server_name.upper())
        else:
            server_list = ServerRegistryInfo.select()
        return [server for server in server_list]

    @classmethod
    @DB.connection_context()
    def load_server_info_from_db(cls):
        for server in cls.query_server_info_from_db():
            server_info = {
                "host": server.f_host,
                "port": server.f_port,
                "protocol": server.f_protocol
            }
            setattr(cls, server.f_server_name.upper(), server_info)


    @classmethod
    @DB.connection_context()
    def save_server_info_to_db(cls, server_name, host, port, protocol="http"):
        server_info = {
            "f_server_name": server_name,
            "f_host": host,
            "f_port": port,
            "f_protocol": protocol
        }
        entity_model, status = ServerRegistryInfo.get_or_create(
            f_server_name=server_name,
            defaults=server_info)
        if status is False:
            for key in server_info:
                setattr(entity_model, key, server_info[key])
            entity_model.save(force_insert=False)

```

##### 首先会调用load_server_info_from_conf方法，加载配置文件中的配置信息。

`SERVICE_CONF`是`service_conf.yaml`
这里导入的配置文件路径是`$FATE_PROJECT_PATH/conf/service_conf.yaml`，还可以导入`$FATE_PROJECT_PATH/conf/local.service_conf.yaml`，并且会覆盖`service_conf.yaml`中相同的配置信息。

<!-- TODO 列出读取的配置信息以及对应的作用 -->

##### 随后会调用load_server_info_from_db方法，加载数据库中的配置信息。

<!-- 还不清楚读取的内容作用是什么 -->

会从`t_server_registry_info`表中读取配置信息，目前可知这个表的每一条记录对应一个服务的配置信息，包括服务名、主机、端口和协议。

```sql
sqlite> PRAGMA table_info(t_server_registry_info);
0|id|INTEGER|1||1
1|f_create_time|INTEGER|0||0
2|f_create_date|DATETIME|0||0
3|f_update_time|INTEGER|0||0
4|f_update_date|DATETIME|0||0
5|f_server_name|VARCHAR(30)|1||0
6|f_host|VARCHAR(30)|1||0
7|f_port|INTEGER|1||0
8|f_protocol|VARCHAR(10)|1||0
```

####  然后再调用ResourceManager.initialize()方法，初始化资源管理器。

ResourceManager来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/manager/resource_manager.py`，只有一个类方法initialize，用于初始化资源管理器。

```python
class ResourceManager(object):
    @classmethod
    def initialize(cls):
        engines_config, engine_group_map = engine_utils.get_engines_config_from_conf(group_map=True)
        for engine_type, engine_configs in engines_config.items():
            for engine_name, engine_config in engine_configs.items():
                cls.register_engine(engine_type=engine_type, engine_name=engine_name, engine_entrance=engine_group_map[engine_type][engine_name], engine_config=engine_config)

    @classmethod
    @DB.connection_context()
    def register_engine(cls, engine_type, engine_name, engine_entrance, engine_config):
        nodes = engine_config.get("nodes", 1)
        cores = engine_config.get("cores_per_node", 0) * nodes * JobDefaultConfig.total_cores_overweight_percent
        memory = engine_config.get("memory_per_node", 0) * nodes * JobDefaultConfig.total_memory_overweight_percent
        filters = [EngineRegistry.f_engine_type == engine_type, EngineRegistry.f_engine_name == engine_name]
        resources = EngineRegistry.select().where(*filters)
        if resources:
            resource = resources[0]
            update_fields = {}
            update_fields[EngineRegistry.f_engine_config] = engine_config
            update_fields[EngineRegistry.f_cores] = cores
            update_fields[EngineRegistry.f_memory] = memory
            update_fields[EngineRegistry.f_remaining_cores] = EngineRegistry.f_remaining_cores + (
                    cores - resource.f_cores)
            update_fields[EngineRegistry.f_remaining_memory] = EngineRegistry.f_remaining_memory + (
                    memory - resource.f_memory)
            update_fields[EngineRegistry.f_nodes] = nodes
            operate = EngineRegistry.update(update_fields).where(*filters)
            update_status = operate.execute() > 0
            if update_status:
                stat_logger.info(f"update {engine_type} engine {engine_name} {engine_entrance} registration information")
            else:
                stat_logger.info(f"update {engine_type} engine {engine_name} {engine_entrance} registration information takes no effect")
        else:
            resource = EngineRegistry()
            resource.f_create_time = base_utils.current_timestamp()
            resource.f_engine_type = engine_type
            resource.f_engine_name = engine_name
            resource.f_engine_entrance = engine_entrance
            resource.f_engine_config = engine_config

            resource.f_cores = cores
            resource.f_memory = memory
            resource.f_remaining_cores = cores
            resource.f_remaining_memory = memory
            resource.f_nodes = nodes
            try:
                resource.save(force_insert=True)
            except Exception as e:
                stat_logger.warning(e)
            stat_logger.info(f"create {engine_type} engine {engine_name} {engine_entrance} registration information")
```

这里只记录了一部分ResourceManager的代码。可知，它会从配置文件中读取引擎配置信息，然后将引擎注册到数据库中。

经过动态调试得到的engines_config, engine_group_map
    
```python
engines_config = 
{'computing': {'EGGROLL': {'cores_per_node': 16, 'nodes': 1},
               'LINKIS_SPARK': {'cores_per_node': 20,
                                'host': '127.0.0.1',
                                'nodes': 2,
                                'port': 9001,
                                'python_path': '/data/projects/fate/python',
                                'token_code': 'MLSS'},
               'SPARK': {'cores_per_node': 20, 'home': None, 'nodes': 2},
               'STANDALONE': {'cores_per_node': 20, 'nodes': 1}},
 'federation': {'EGGROLL': {'host': '127.0.0.1', 'port': 9370},
                'PULSAR': {'cluster': 'standalone',
                           'host': '192.168.0.5',
                           'max_message_size': 1048576,
                           'mng_port': 8080,
                           'mode': 'replication',
                           'port': 6650,
                           'route_table': None,
                           'tenant': 'fl-tenant',
                           'topic_ttl': 30},
                'RABBITMQ': {'host': '192.168.0.4',
                             'max_message_size': 1048576,
                             'mng_port': 12345,
                             'mode': 'replication',
                             'password': 'fate',
                             'port': 5672,
                             'route_table': None,
                             'user': 'fate'},
                'STANDALONE': {'cores_per_node': 20, 'nodes': 1}},
 'storage': {'EGGROLL': {'cores_per_node': 16, 'nodes': 1},
             'HDFS': {'name_node': 'hdfs://fate-cluster', 'path_prefix': None},
             'HIVE': {'auth_mechanism': None,
                      'host': '127.0.0.1',
                      'password': None,
                      'port': 10000,
                      'username': None},
             'LINKIS_HIVE': {'host': '127.0.0.1', 'port': 9001},
             'STANDALONE': {'cores_per_node': 20, 'nodes': 1}}}

engine_group_map = 
{'computing': {'EGGROLL': 'fate_on_eggroll',
               'LINKIS_SPARK': 'fate_on_spark',
               'SPARK': 'fate_on_spark',
               'STANDALONE': 'fate_on_standalone'},
 'federation': {'EGGROLL': 'fate_on_eggroll',
                'PULSAR': 'fate_on_spark',
                'RABBITMQ': 'fate_on_spark',
                'STANDALONE': 'fate_on_standalone'},
 'storage': {'EGGROLL': 'fate_on_eggroll',
             'HDFS': 'fate_on_spark',
             'HIVE': 'fate_on_spark',
             'LINKIS_HIVE': 'fate_on_spark',
             'STANDALONE': 'fate_on_standalone'}}

```

可知相关资源引擎分成三类：计算引擎、联邦引擎和存储引擎，分别对应`computing`、`federation`和`storage`三个类别，每个类别下面有多个引擎，比如`computing`类别下面有`EGGROLL`、`LINKIS_SPARK`、`SPARK`和`STANDALONE`四个引擎。

#### 最后调用RuntimeConfig.load_config_manager()方法，初始化环境变量。

`RuntimeConfig`是`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/db/runtime_config.py`中的一个静态类，它的load_config_manager方法会将LOAD_CONFIG_MANAGER设置为True。
代表已经加载了配置管理器。

```python
    @classmethod
    def load_config_manager(cls):
        cls.LOAD_CONFIG_MANAGER = True
```

### 随后调用RuntimeConfig.init_env()方法，初始化环境变量。

```python
    @classmethod
    def init_env(cls):
        cls.ENV.update(get_versions())
```

这个操作会在RuntimeConfig的ENV属性中添加版本信息。

### 随后调用RuntimeConfig.init_config()方法，初始化配置信息。

```python
    @classmethod
    def init_config(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
```

`RuntimeConfig.init_config(JOB_SERVER_HOST=HOST, HTTP_PORT=HTTP_PORT)`
在这里会给RuntimeConfig的JOB_SERVER_HOST和HTTP_PORT属性赋值。

### 随后调用RuntimeConfig.set_process_role()方法，设置进程角色。

```python
    @classmethod
    def set_process_role(cls, process_role: ProcessRole):
        cls.PROCESS_ROLE = process_role
```
其实就是给RuntimeConfig的PROCESS_ROLE属性赋值为字符串"driver"

### 随后调用RuntimeConfig.set_service_db()方法，设置服务数据库。
```python
    RuntimeConfig.set_service_db(service_db())
    RuntimeConfig.SERVICE_DB.register_flow()
    RuntimeConfig.SERVICE_DB.register_models()
```


```python
    @classmethod
    def set_service_db(cls, service_db):
        cls.SERVICE_DB = service_db
```

`service_db()`来自`$FATE_FLOW_PYTHON_FATE_FLOW_PACKAGE/db/db_services.py`

```python
def service_db():
    """Initialize services database.
    Currently only ZooKeeper is supported.

    :return ZooKeeperDB if `use_registry` is `True`, else FallbackDB.
            FallbackDB is a compatible class and it actually does nothing.
    """
    if not USE_REGISTRY:
        return FallbackDB()
    if isinstance(USE_REGISTRY, str):
        if USE_REGISTRY.lower() == 'zookeeper':
            return ZooKeeperDB()
    # backward compatibility
    return ZooKeeperDB()
```

这里会根据USE_REGISTRY的值来决定返回什么类型的数据库，USE_REGISTRY的值是从`$FATE_PROJECT_PATH/conf/service_conf.yaml`中的`use_registry`字段读取的，默认是`false`，所以返回的是`FallbackDB`，它是一个兼容类，实际上什么都不做。

`ZooKeeperDB`本质上是一个开源的分布式协调服务，它的作用是用来存储服务的配置信息，大概就是一个分布式的数据库，fate用来注册和储存flow server的地址和训练好的模型的下载地址。


### 至此，基本配置信息初始化完成，RuntimeConfig还会记录一些其他的信息，例如组件提供者，这些信息会在后续初始化组件的时候再进行初始化。