# 关于reader组件output_data的问题

## 问题

在《FATE框架组件运行详细研究》中

发现reader组件运行执行完后记录的日志显示`output data`为空

Reader

```python
"""
[INFO] [2023-10-28 14:29:53,598] [202310281429459592480] [528651:139890981195776] - [task_executor._run_] [line:214]: task output dsl {'data': ['data']}
[INFO] [2023-10-28 14:29:53,599] [202310281429459592480] [528651:139890981195776] - [task_executor._run_] [line:215]: task output data [None]
"""
```

对比其他组件

data_transform

```python
"""
[INFO] [2023-10-28 14:30:00,219] [202310281429459592480] [528759:139625712803840] - [task_executor._run_] [line:214]: task output dsl {'data': ['data'], 'model': ['model']}
[INFO] [2023-10-28 14:30:00,219] [202310281429459592480] [528759:139625712803840] - [task_executor._run_] [line:215]: task output data [<fate_arch.computing.standalone._table.Table object at 0x7efd15095610>]
"""
```

而在fateboard中reader组件的详细内容中的data_output页面下又可以看到

![reader组件_data_output](.\images\reader组件_data_output.png)

## 分析

### `output data`为空

首先经过代码分析可知产生日志的代码

task_executor.\_run\_

```python
            ...
            ...
            ...
			if profile_log_enabled:
                # add profile logs
                LOGGER.info("profile logging is enabled")
                profile.profile_start()
                cpn_output = run_object.run(cpn_input)
                sess.wait_remote_all_done()
                profile.profile_ends()
            else:
                LOGGER.info("profile logging is disabled")
                cpn_output = run_object.run(cpn_input)	# 跟进cpn_output
                sess.wait_remote_all_done()

            LOGGER.info(f"task output dsl {task_output_dsl}")
            LOGGER.info(f"task output data {cpn_output.data}")
            ...
            ...
            ...
```

跟进cpn_output

ComponentBase.run

```python
    def run(self, cpn_input: ComponentInputProtocol, retry: bool = True):
        self.task_version_id = cpn_input.task_version_id
        self.tracker = cpn_input.tracker
        self.checkpoint_manager = cpn_input.checkpoint_manager

        # retry
        if (
            retry
            and hasattr(self, '_retry')
            and callable(self._retry)
            and self.checkpoint_manager is not None
            and self.checkpoint_manager.latest_checkpoint is not None
        ):
            self._retry(cpn_input=cpn_input)
        # normal
        else:
            self._run(cpn_input=cpn_input)

        return ComponentOutput(data=self.save_data(), models=self.export_model(), cache=self.save_cache(), serialize=self.serialize)

    def save_data(self):
        return self.data_output
```

可知`cpn_output.data`就是组件对象的`data_output`成员变量

根据此字段搜索，发现在reader组件的执行过程中，没有任何一处地方修改了`data_output`属性，这就解释了为什么reader的日志显示`output data`为空

那么为什么fateboard中可以查询到输出的数据表呢？

### fateboard中可以看见输出数据

定位一下网络请求

![reader组件data请求跟踪](.\images\reader组件data请求跟踪.png)

可知fateboard在查看组件输出数据的时候请求了`v1/tracking/component/output/data`这个api，结合负载以及响应数据可以确定是这个请求获取了数据

自然可以定位到${fateflow目录}/python/fate_flow/app包下的tracking_app.py

定位到component_output_data

```python
@manager.route('/component/output/data', methods=['post'])
def component_output_data():
    request_data = request.json
    tasks = JobSaver.query_task(only_latest=True, job_id=request_data['job_id'],
                                component_name=request_data['component_name'],
                                role=request_data['role'], party_id=request_data['party_id'])   # 根据job_id, component_name, role, party_id查询相应的task
    if not tasks:
        raise ValueError(f'no found task, please check if the parameters are correct:{request_data}')
    import_component_output_depend(tasks[0].f_provider_info)    # 导入组件依赖，其实就是添加了sys.path
    output_tables_meta = get_component_output_tables_meta(task_data=request_data)   # 根据request_data获取组件输出的表的元数据
    if not output_tables_meta:
        return get_json_result(retcode=0, retmsg='no data', data=[])    # 如果没有查找到数据，返回空列表
    output_data_list = []   
    headers = []
    totals = []
    data_names = []
    for output_name, output_table_meta in output_tables_meta.items():   # 遍历查找到的表的元数据
        output_data = []
        is_str = False
        all_extend_header = {}
        if output_table_meta:   # 如果表的元数据不为空，即生成了对象，查找到了数据
            for k, v in output_table_meta.get_part_of_data():   # 遍历表的数据
                data_line, is_str, all_extend_header = feature_utils.get_component_output_data_line(src_key=k, src_value=v, schema=output_table_meta.get_schema(), all_extend_header=all_extend_header)
                output_data.append(data_line)   # 将数据添加到output_data中
            total = output_table_meta.get_count()   # 获取表的数据总数
            output_data_list.append(output_data)    # 将output_data添加到output_data_list中
            data_names.append(output_name)  # 将output_name添加到data_names中
            totals.append(total)    # 将total添加到totals中
        if output_data:  # 如果output_data不为空，即在这张表中查找到了数据
            extend_header = feature_utils.generate_header(all_extend_header, schema=output_table_meta.get_schema())  # 生成header
            if output_table_meta.schema.get("is_display", True):    
                header = get_component_output_data_schema(output_table_meta=output_table_meta, is_str=is_str,
                                                          extend_header=extend_header)  
            else:
                header = [] # 如果schema中的is_display为False，即不显示，header为空列表
            headers.append(header)  # 将header添加到headers中
        else:
            headers.append(None)    # 如果output_data为空，即在这张表中没有查找到数据，header为空
    if len(output_data_list) == 1 and not output_data_list[0]:  # 如果output_data_list中只有一个元素且为空
        return get_json_result(retcode=0, retmsg='no data', data=[])    # 返回空列表
    return get_json_result(retcode=0, retmsg='success', data=output_data_list,
                           meta={'header': headers, 'total': totals, 'names': data_names})
```

其中可以知道`output_data_list`是最后返回数据的列表，生成`output_data_list`的过程跟`output_tables_meta`有关

`output_tables_meta`的生成

```python
output_tables_meta = get_component_output_tables_meta(task_data=request_data)   # 根据request_data获取组件输出的表的元数据
```

get_component_output_tables_meta

```python
def get_component_output_tables_meta(task_data):
    check_request_parameters(task_data) # 检查请求参数
    tracker = Tracker(job_id=task_data['job_id'], component_name=task_data['component_name'],
                      role=task_data['role'], party_id=task_data['party_id'])   # 根据请求参数创建Tracker对象
    output_data_table_infos = tracker.get_output_data_info()    # 获取组件输出的表的信息
    output_tables_meta = tracker.get_output_data_table(output_data_infos=output_data_table_infos)   # 根据组件输出的表的信息获取表的元数据
    return output_tables_meta

```

其中，Tracker对象的生成是静态的属性标注，意味着创建对象的过程中除了给属性复制不会执行有很大影响的代码，但是在方法`tracker.get_output_data_info`方法中，一路跟进

```python
    def get_output_data_info(self, data_name=None):
        return self.read_output_data_info_from_db(data_name=data_name)
    
    def read_output_data_info_from_db(self, data_name=None):
        filter_dict = {}
        filter_dict["job_id"] = self.job_id
        filter_dict["component_name"] = self.component_name
        filter_dict["role"] = self.role
        filter_dict["party_id"] = self.party_id
        if data_name:
            filter_dict["data_name"] = data_name
        return self.query_output_data_infos(**filter_dict)

    @classmethod
    @DB.connection_context()
    def query_output_data_infos(cls, **kwargs) -> typing.List[TrackingOutputDataInfo]:
        try:
            tracking_output_data_info_model = cls.get_dynamic_db_model(TrackingOutputDataInfo, kwargs.get("job_id"))
            filters = []
            for f_n, f_v in kwargs.items():
                attr_name = 'f_%s' % f_n
                if hasattr(tracking_output_data_info_model, attr_name):
                    filters.append(operator.attrgetter('f_%s' % f_n)(tracking_output_data_info_model) == f_v)
            if filters:
                output_data_infos_tmp = tracking_output_data_info_model.select().where(*filters)
            else:
                output_data_infos_tmp = tracking_output_data_info_model.select()
            output_data_infos_group = {}
            # only the latest version of the task output data is retrieved
            for output_data_info in output_data_infos_tmp:
                group_key = cls.get_output_data_group_key(output_data_info.f_task_id, output_data_info.f_data_name)
                if group_key not in output_data_infos_group:
                    output_data_infos_group[group_key] = output_data_info
                elif output_data_info.f_task_version > output_data_infos_group[group_key].f_task_version:
                    output_data_infos_group[group_key] = output_data_info
            return list(output_data_infos_group.values())
        except Exception as e:
            return []
```

可知`tracker.get_output_data_info`方法实现的功能就是在`TrackingOutputDataInfo`表中查找满足`job_id`、`component_name`、`role`和`party_id`还有`data_name`（如果有）的记录，然后遍历找到的记录，最后筛选出`task_version`字段最大的记录（最新的记录）返回。

也就是说`output_data_table_infos`是一个列表，列表中的元素都是`TrackingOutputDataInfo`的实例，然后由`output_tables_meta`（我们关注的结果）的生成过程

```python
output_tables_meta = tracker.get_output_data_table(output_data_infos=output_data_table_infos)   # 根据组件输出的表的信息获取表的元数据
```

跟进tracker.get_output_data_table

```PYTHON
    def get_output_data_table(self, output_data_infos, tracker_client=None):
        """
        Get component output data table, will run in the task executor process
        :param output_data_infos:
        :return:
        """
        output_tables_meta = {}
        if output_data_infos:
            for output_data_info in output_data_infos:  # 遍历所有输出表的信息
                schedule_logger(self.job_id).info("get task {} {} output table {} {}".format(output_data_info.f_task_id, output_data_info.f_task_version, output_data_info.f_table_namespace, output_data_info.f_table_name))
                if not tracker_client:
                    data_table_meta = storage.StorageTableMeta(name=output_data_info.f_table_name, namespace=output_data_info.f_table_namespace)    # 根据表名和命名空间创建StorageTableMeta对象，在这个过程中会从数据库中获取表的元信息	（跟进）
                else:
                    data_table_meta = tracker_client.get_table_meta(output_data_info.f_table_name, output_data_info.f_table_namespace)  

                output_tables_meta[output_data_info.f_data_name] = data_table_meta  # 将表的元信息存入output_tables_meta字典中
        return output_tables_meta
```

跟进storage.StorageTableMeta

```PYTHON
class StorageTableMeta(StorageTableMetaABC):

    def __init__(self, name, namespace, new=False, create_address=True):
        self.name = name
        self.namespace = namespace
        self.address = None
        self.engine = None
        self.store_type = None
        self.options = None
        self.partitions = None
        self.in_serialized = None
        self.have_head = None
        self.id_delimiter = None
        self.extend_sid = False
        self.auto_increasing_sid = None
        self.schema = None
        self.count = None
        self.part_of_data = None
        self.description = None
        self.origin = None
        self.disable = None
        self.create_time = None
        self.update_time = None
        self.read_access_time = None
        self.write_access_time = None
        if self.options is None:
            self.options = {}
        if self.schema is None:
            self.schema = {}
        if self.part_of_data is None:
            self.part_of_data = []
        if not new:
            self.build(create_address)

    def build(self, create_address):
        for k, v in self.table_meta.__dict__["__data__"].items():
            setattr(self, k.lstrip("f_"), v)
        if create_address:
            self.address = self.create_address(storage_engine=self.engine, address_dict=self.address)

    def __new__(cls, *args, **kwargs):  # 创建对象时发生，在__init__之前
        if not kwargs.get("new", False):    # 如果是查询表（非新建）
            name, namespace = kwargs.get("name"), kwargs.get("namespace")
            if not name or not namespace:
                return None
            tables_meta = cls.query_table_meta(filter_fields=dict(name=name, namespace=namespace))  # 根据name和namespace在StorageTableMetaModel表中查询
            if not tables_meta:
                return None
            self = super().__new__(cls)
            setattr(self, "table_meta", tables_meta[0]) # 将查询到的表元数据赋值给self.table_meta
            return self
        else:   # 如果是新建表，则直接返回，正常创建对象
            return super().__new__(cls)
```

可知`data_table_meta`是一个StorageTableMeta对象，该对象在生成时会在StorageTableMetaModel表中根据表名和数据库名查找记录，如果查找到了，就将此纪录（list类型）赋值给`table_meta`属性。

综上`output_tables_meta`是一个字典，键是数据名，值是对应的数据表元信息`StorageTableMeta`对象。

让我们回到component_output_data函数中从`output_tables_meta`获取数据并返回的位置

```python
    output_data_list = []   
    headers = []
    totals = []
    data_names = []
    for output_name, output_table_meta in output_tables_meta.items():   # 遍历查找到的表的元数据
        output_data = []
        is_str = False
        all_extend_header = {}
        if output_table_meta:   # 如果表的元数据不为空，即生成了对象，查找到了数据
            for k, v in output_table_meta.get_part_of_data():   # 遍历表的数据
                data_line, is_str, all_extend_header = feature_utils.get_component_output_data_line(src_key=k, src_value=v, schema=output_table_meta.get_schema(), all_extend_header=all_extend_header)
                output_data.append(data_line)   # 将数据添加到output_data中
            total = output_table_meta.get_count()   # 获取表的数据总数
            output_data_list.append(output_data)    # 将output_data添加到output_data_list中
            data_names.append(output_name)  # 将output_name添加到data_names中
            totals.append(total)    # 将total添加到totals中
        if output_data:  # 如果output_data不为空，即在这张表中查找到了数据
            extend_header = feature_utils.generate_header(all_extend_header, schema=output_table_meta.get_schema())  # 生成header
            if output_table_meta.schema.get("is_display", True):    
                header = get_component_output_data_schema(output_table_meta=output_table_meta, is_str=is_str,
                                                          extend_header=extend_header)  
            else:
                header = [] # 如果schema中的is_display为False，即不显示，header为空列表
            headers.append(header)  # 将header添加到headers中
        else:
            headers.append(None)    # 如果output_data为空，即在这张表中没有查找到数据，header为空
    if len(output_data_list) == 1 and not output_data_list[0]:  # 如果output_data_list中只有一个元素且为空
        return get_json_result(retcode=0, retmsg='no data', data=[])    # 返回空列表
    return get_json_result(retcode=0, retmsg='success', data=output_data_list,
                           meta={'header': headers, 'total': totals, 'names': data_names})
```

可知output_data_list是一个列表，列表中的元素是每一个output_data（从每一个前面获取到的输出表元数据`StorageTableMeta`对象中通过`get_part_of_data`方法获取的数据）

StorageTableMeta.get_part_of_data

```python
    def get_part_of_data(self):
        return self.part_of_data
```

可知获取数据的直接接口就是`StorageTableMeta`对象的`part_of_data`属性，那么查找修改了该属性的值，发现在前面整个查找过程中都没有显式的给`part_of_data`属性赋值，已知生成对象过程会将`table_meta`属性赋值为查找到的`StorageTableMetaModel`D对象，而在`StorageTableMeta.__init__`函数的最后会调用`build`方法

StorageTableMeta.build

```PYTHON
    def build(self, create_address):
        for k, v in self.table_meta.__dict__["__data__"].items():
            setattr(self, k.lstrip("f_"), v)
        if create_address:
            self.address = self.create_address(storage_engine=self.engine, address_dict=self.address)
```

也就是说唯一可能给`part_of_data`属性赋值的地方就在这里了，分析`StorageTableMetaModel`

```PYTHON
class StorageTableMetaModel(DataBaseModel):
    f_name = CharField(max_length=100, index=True)
    f_namespace = CharField(max_length=100, index=True)
    f_address = JSONField()
    f_engine = CharField(max_length=100)  # 'EGGROLL', 'MYSQL'
    f_store_type = CharField(max_length=50, null=True)  # store type
    f_options = JSONField()
    f_partitions = IntegerField(null=True)

    f_id_delimiter = CharField(null=True)
    f_in_serialized = BooleanField(default=True)
    f_have_head = BooleanField(default=True)
    f_extend_sid = BooleanField(default=False)
    f_auto_increasing_sid = BooleanField(default=False)

    f_schema = SerializedField()
    f_count = BigIntegerField(null=True)
    f_part_of_data = SerializedField()
    f_origin = CharField(max_length=50, default='')
    f_disable = BooleanField(default=False)
    f_description = TextField(default='')

    f_read_access_time = BigIntegerField(null=True)
    f_read_access_date = DateTimeField(null=True)
    f_write_access_time = BigIntegerField(null=True)
    f_write_access_date = DateTimeField(null=True)

    class Meta:
        db_table = "t_storage_table_meta"
        primary_key = CompositeKey('f_name', 'f_namespace')
```

果然有`f_part_of_data`属性，并且是做了序列化处理的

```PYTHON
    f_part_of_data = SerializedField()
```

也就是说数据本来就已经存在于数据库中了，并且根据前面的使用方式猜测数据存储在序列化的字典对象中，那么就确定了fateboard中看见的输出数据是从`f_part_of_data`字段中来的，并且该字段在查询之前已经存储在了数据库，现在只需找到在哪个步骤中存储的，就可以验证我们的猜想。

搜索发现`StorageTableMetaModel`只在`StorageTableMeta`类中有使用到，说明后者是前者唯一的操作接口。

回到reader组件_run方法，根据之前基于日志分析的调用链一层一层跟进

```PYTHON
        self.save_table(src_table=input_table, dest_table=output_table) # 将输入表的数据存储到输出表中，其实就是将输入表的数据复制到输出表中，在这一步保存表的过程中就会保存output_table的meta info了，所以后面可以查找到
```

save_table

```PYTHON
        if src_table.engine == dest_table.engine and src_table.meta.get_in_serialized():
            self.to_save(src_table, dest_table)
```

to_save

```PYTHON
        self.tracker.job_tracker.save_output_data(
            src_computing_table,
            output_storage_engine=dest_table.engine,
            output_storage_address=dest_table.address.__dict__,
            output_table_namespace=dest_table.namespace,
            output_table_name=dest_table.name,
            schema=schema,
            need_read=False
        )
```

Tracker.save_output_data

```PYTHON
            part_of_data = []
            if need_read:
                for k, v in computing_table.collect():	# 生成part_of_data的地方
                    part_of_data.append((k, v))
                    part_of_limit -= 1
                    if part_of_limit == 0:
                        break

            session.Session.persistent(computing_table=computing_table,
                                       namespace=output_table_namespace,
                                       name=output_table_name,
                                       schema=schema,
                                       part_of_data=part_of_data,
                                       engine=output_storage_engine,
                                       engine_address=output_storage_address,
                                       token=token)
```

Session.persistent

```PYTHON
        return StorageSessionBase.persistent(computing_table=computing_table,
                                             namespace=namespace,
                                             name=name,
                                             schema=schema,
                                             part_of_data=part_of_data,
                                             engine=engine,
                                             engine_address=engine_address,
                                             store_type=store_type,
                                             token=token)

```

StorageSessionBase.persistent

```PYTHON
        address = StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)
        schema = schema if schema else {}
        computing_table.save(address, schema=schema, partitions=partitions, store_type=store_type)
        table_count = computing_table.count()
        table_meta = StorageTableMeta(name=name, namespace=namespace, new=True)
        table_meta.address = address
        table_meta.partitions = computing_table.partitions
        table_meta.engine = engine
        table_meta.store_type = store_type
        table_meta.schema = schema
        table_meta.part_of_data = part_of_data if part_of_data else {}
        table_meta.count = table_count
        table_meta.write_access_time = current_timestamp()
        table_meta.origin = StorageTableOrigin.OUTPUT
        table_meta.create()	# 创建表的元信息记录
```

找到了生成part_of_data，以及创建`StorageTableMetaModel`表的地方，猜想得证！但是`f_part_of_data`是一个列表，列表中的值都是键值对元组，详细的就要追溯`_Table`类的`collect`方法了。

<hr>

## 总结

`output data`为空是因为reader组件在执行过程中没有修改与之相关的组件属性`data_output`

而fateboard中看见的输出数据是直接走的组件输出数据查询接口，该接口会直接在数据库中首先查找相关作业，组件和角色等所对应的任务，然后根据接口请求信息（`job_id,component_name,role,party_id`等）从数据库中查找组件输出数据元数据。

我的理解中`meta`所代表的元数据，应该是指真实数据的属性信息，也就是一些与数据无关的，类似记录数据所属作业，任务，角色，创建修改时间等的信息，就像键值对中的键一样，具有索引的功能，但是这里查找的`组件输出数据元数据`还包括了`f_part_of_data`字段，这个字段的意思是部分数据，也就是说这个接口只是一个预览数据的接口，相关的记录只能查询到部分的数据，储存该部分数据的过程中规定部分数据量大小的地方在`Tracker.save_output_data`

```PYTHON
            part_of_limit = JobDefaultConfig.output_data_summary_count_limit    # 读取JobDefaultConfig中最大的输出部分数据的行数
            part_of_data = []
            if need_read:
                for k, v in computing_table.collect():
                    part_of_data.append((k, v))
                    part_of_limit -= 1
                    if part_of_limit == 0:
                        break
```

`JobDefaultConfig.output_data_summary_count_limit`的默认值是`100`

也就是说fateboard预览的是一个组件存储输出数据的“记录”，这个预览界面只能查看默认`100`条数据，从查找到获取数据的过程跟reader组件执行过程的`data_output`无关，并且这一部分预览的数据只有一部分的真实数据，或者说真实数据的元信息（**summary**（概要）），类似实现了预览缓存的功能。

那么真实的全部数据在哪里呢？

由reader组件的执行流程，找到记录缓存的地方

### 真实数据的保存过程

回到

StorageSessionBase.persistent

```PYTHON
        address = StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)
        schema = schema if schema else {}
        computing_table.save(address, schema=schema, partitions=partitions, store_type=store_type)	# 全部计算数据保存
        table_count = computing_table.count()
        table_meta = StorageTableMeta(name=name, namespace=namespace, new=True)
        table_meta.address = address
        table_meta.partitions = computing_table.partitions
        table_meta.engine = engine
        table_meta.store_type = store_type
        table_meta.schema = schema
        table_meta.part_of_data = part_of_data if part_of_data else {}
        table_meta.count = table_count
        table_meta.write_access_time = current_timestamp()
        table_meta.origin = StorageTableOrigin.OUTPUT
        table_meta.create()	# 元信息记录保存
```

经过再深层的跟进

追溯`computing_table`

reader.to_save

```PYTHON
        src_computing_table = session.get_computing_session().load(
            src_table_meta.get_address(),
            schema=src_table_meta.get_schema(),
            partitions=src_table_meta.get_partitions(),
            id_delimiter=src_table_meta.get_id_delimiter(),
            in_serialized=src_table_meta.get_in_serialized(),
        )   # 根据输入表的meta info获得一个计算表
```

standalone._csession.CSession.load

```python
    def load(self, address: AddressABC, partitions: int, schema: dict, **kwargs):
        from fate_arch.common.address import StandaloneAddress
        from fate_arch.storage import StandaloneStoreType

        if isinstance(address, StandaloneAddress):
            raw_table = self._session.load(address.name, address.namespace)
            if address.storage_type != StandaloneStoreType.ROLLPAIR_IN_MEMORY:
                raw_table = raw_table.save_as(
                    name=f"{address.name}_{fate_uuid()}",
                    namespace=address.namespace,
                    partition=partitions,
                    need_cleanup=True,
                )
            table = Table(raw_table)
            table.schema = schema
            return table

        from fate_arch.common.address import PathAddress

        if isinstance(address, PathAddress):
            from fate_arch.computing.non_distributed import LocalData
            from fate_arch.computing import ComputingEngine
            return LocalData(address.path, engine=ComputingEngine.STANDALONE)
        raise NotImplementedError(
            f"address type {type(address)} not supported with standalone backend"
        )
```

`session.get_computing_session()`返回一个实现了`CSessionABC`的计算会话对象，在单机版中是`standalone._csession.CSession`对象，该计算会话类对象有一个属性为`_session`，是一个`fate_arch._standalone.Session`对象，该对象没有实现接口，其中

fate_arch._standalone.Session.load

```PYTHON
    def load(self, name, namespace):
        return _load_table(session=self, name=name, namespace=namespace)
```

fate_arch.\_standalone.Session._load\_table

```python
def _load_table(session, name, namespace, need_cleanup=False):
    partitions = _TableMetaManager.get_table_meta(namespace, name)
    if partitions is None:
        raise RuntimeError(f"table not exist: name={name}, namespace={namespace}")
    return Table(
        session=session,
        namespace=namespace,
        name=name,
        partitions=partitions,
        need_cleanup=need_cleanup,
    )
```

而回到`standalone._csession.CSession.load`中，可知`raw_table`是一个`fate_arch._standalone.Table`对象，该对象也没有实现接口，`table`是`fate_arch.computing.standalone._table.Table`对象，继承了`CTableABC`类，包含属性`_table`，就是在构造对象时传入的`raw_table`对象

fate_arch._standalone.Table

```PYTHON
class Table(object):
    def __init__(
        self,
        session: "Session",
        namespace: str,
        name: str,
        partitions,
        need_cleanup=True,
    ):
        self._need_cleanup = need_cleanup
        self._namespace = namespace
        self._name = name
        self._partitions = partitions
        self._session = session
```

fate_arch.computing.standalone._table.Table

```python
class Table(CTableABC):
    def __init__(self, table):
        self._table = table
        self._engine = ComputingEngine.STANDALONE

        self._count = None
```



可知`computing_table`就是一个计算表对象，在`reader.save_table`之前只是声明表，现在追溯保存表的过程，回到

StorageSessionBase.persistent

```PYTHON
        computing_table.save(address, schema=schema, partitions=partitions, store_type=store_type)
        table_count = computing_table.count()
```

`computing_table`是`CTableABC`（一个抽象类，可看作是计算表接口）的子类，单机版中可以看作是到`fate_arch.computing.standalone._table`中的`Table`类

Table.save

```PYTHON
    @computing_profile
    def save(self, address, partitions, schema, **kwargs):
        from fate_arch.common.address import StandaloneAddress

        if isinstance(address, StandaloneAddress):
            self._table.save_as(
                name=address.name,
                namespace=address.namespace,
                partition=partitions,
                need_cleanup=False,
            )
            schema.update(self.schema)
            return

        from fate_arch.common.address import PathAddress

        if isinstance(address, PathAddress):
            from fate_arch.computing.non_distributed import LocalData

            return LocalData(address.path)
        raise NotImplementedError(
            f"address type {type(address)} not supported with standalone backend"
        )
```

前面已知`Table`类含有的属性`_table`是`fate_arch._standalone.Table`对象

fate_arch._standalone.Table.save_as

```PYTHON
    def save_as(self, name, namespace, partition=None, need_cleanup=True):
        if partition is None:
            partition = self._partitions
        # noinspection PyProtectedMember
        dup = _create_table(self._session, name, namespace, partition, need_cleanup)
        dup.put_all(self.collect())
        return dup
```

fate_arch._standalone.\_create_table

```PYTHON
def _create_table(
    session: "Session",
    name: str,
    namespace: str,
    partitions: int,
    need_cleanup=True,
    error_if_exist=False,
):
    assert isinstance(namespace, str)
    assert isinstance(name, str)
    assert isinstance(partitions, int)

    exist_partitions = _TableMetaManager.get_table_meta(namespace, name)
    if exist_partitions is None:
        _TableMetaManager.add_table_meta(namespace, name, partitions)
    else:
        if error_if_exist:
            raise RuntimeError(f"table already exist: name={name}, namespace={namespace}")
        partitions = exist_partitions

    return Table(
        session=session,
        namespace=namespace,
        name=name,
        partitions=partitions,
        need_cleanup=need_cleanup,
    )
```

返回一个`fate_arch._standalone.Table`对象

fate_arch._standalone.Table.put_all

```PYTHON
    def put_all(self, kv_list: Iterable):
        txn_map = {}
        is_success = True
        with ExitStack() as s:
            for p in range(self._partitions):
                env = s.enter_context(self._get_env_for_partition(p, write=True))
                txn_map[p] = env, env.begin(write=True)
            for k, v in kv_list:
                try:
                    k_bytes, v_bytes = _kv_to_bytes(k=k, v=v)
                    p = _hash_key_to_partition(k_bytes, self._partitions)
                    is_success = is_success and txn_map[p][1].put(k_bytes, v_bytes)
                except Exception as e:
                    is_success = False
                    LOGGER.exception(f"put_all for k={k} v={v} fail. exception: {e}")
                    break
            for p, (env, txn) in txn_map.items():
                txn.commit() if is_success else txn.abort()
```

可知`dup`是一个临时的`fate_arch._standalone.Table`对象，`fate_arch._standalone.Table`类的很多方法都是底层跟数据库的操作过程，在这里就是使用的python`lmdb`库操作数据库，`put_all`方法将传入的数据全部写入数据库，这里以及是最底层的操作了，再往下就是`lmdb`库的内容了。

同理，再调用`fate_arch._standalone.Table.put_all`的时候传入的参数由`collect`方法提供

fate_arch._standalone.Table.collect

```PYTHON
    def collect(self, **kwargs):
        iterators = []
        with ExitStack() as s:
            for p in range(self._partitions):
                env = s.enter_context(self._get_env_for_partition(p))
                txn = s.enter_context(env.begin())
                iterators.append(s.enter_context(txn.cursor()))

            # Merge sorted
            entries = []
            for _id, it in enumerate(iterators):
                if it.next():
                    key, value = it.item()
                    entries.append([key, value, _id, it])
            heapify(entries)
            while entries:
                key, value, _, it = entry = entries[0]
                yield deserialize(key), deserialize(value)
                if it.next():
                    entry[0], entry[1] = it.item()
                    heapreplace(entries, entry)
                else:
                    _, _, _, it = heappop(entries)
```

这个就是很直接的以迭代器的形式将数据库中的数据读出返回。而数据库中的数据在上传阶段已经由`upload`组件写入了，所以这里可以获取到。

注意，以上的`computing_table`是`src_computing_table`，也就是reader从数据库中读取的数据，而

```python
        self.tracker.job_tracker.save_output_data(  # 保存输出表
            src_computing_table,
            output_storage_engine=dest_table.engine,
            output_storage_address=dest_table.address.__dict__,
            output_table_namespace=dest_table.namespace,
            output_table_name=dest_table.name,
            schema=schema,
            need_read=False
        )
```

是通过`src_computing_table`的`save`方法（由方法分析可以理解为实现了另存为功能），在

```PYTHON
        computing_table.save(address, schema=schema, partitions=partitions, store_type=store_type)
```

这一部分中将读取到的源数据表另存为了`name`和`namespace`等信息不同，但数据内容一致的输出表，在这里这些信息由address决定

追溯`address`

```python
### 
address = StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)


### 
if engine == StorageEngine.STANDALONE:
    address_dict.update({"name": name, "namespace": namespace})
    store_type = StandaloneStoreType.ROLLPAIR_LMDB if store_type is None else store_type

###
if engine_address is None:
    # find engine address from service_conf.yaml
    engine_address = engine_utils.get_engines_config_from_conf().get(EngineType.STORAGE, {}).get(engine, {})
address_dict = engine_address.copy()	# 说白了address_dict就是个字典
```

得证



## 额外

由上可知`Tracker.save_output_data`其实是将计算表保存到数据的一个接口，其实通过代码分析也可知在`TaskExecutor.\_run\_`中，组件执行完任务后，也会进行数据的保存

```PYTHON
            if profile_log_enabled:
                # add profile logs
                LOGGER.info("profile logging is enabled")
                profile.profile_start()
                cpn_output = run_object.run(cpn_input)
                sess.wait_remote_all_done()
                profile.profile_ends()
            else:
                LOGGER.info("profile logging is disabled")
                cpn_output = run_object.run(cpn_input)  # 进入组件的run方法，开始执行组件
                sess.wait_remote_all_done()

            LOGGER.info(f"task output dsl {task_output_dsl}")
            LOGGER.info(f"task output data {cpn_output.data}")

            output_table_list = []
            for index, data in enumerate(cpn_output.data):
                data_name = task_output_dsl.get('data')[index] if task_output_dsl.get('data') else '{}'.format(index)
                #todo: the token depends on the engine type, maybe in job parameters
                persistent_table_namespace, persistent_table_name = tracker.save_output_data(
                    computing_table=data,
                    output_storage_engine=job_parameters.storage_engine,
                    token={"username": user_name})  # 将组件的输出结果（计算表的形式）保存到数据库中
                if persistent_table_namespace and persistent_table_name:
                    tracker.log_output_data_info(data_name=data_name,
                                                 table_namespace=persistent_table_namespace,
                                                 table_name=persistent_table_name)  # 将组件的输出结果（计算表的形式）的信息（表名，命名空间，等记录）保存到数据库中
                    output_table_list.append({"namespace": persistent_table_namespace, "name": persistent_table_name})  # 猜测这个可能会在下一个组件获取输入的时候用到
            self.log_output_data_table_tracker(args.job_id, input_table_list, output_table_list)    # 将组件的输入输出表的信息保存到数据库中
```

<!--TODO: 将上面过程重新梳理，涉及到的每个类和类方法以及相互关系做都做一个UML图来展示，完成整个流程架构图 -->

