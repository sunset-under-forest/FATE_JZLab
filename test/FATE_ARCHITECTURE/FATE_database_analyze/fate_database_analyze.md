# fate flow server 数据库

## 数据库初始化，建立连接过程分析

入口`$FATE_PROJECT_PATH/fate_flow/fate_flow_server.py`

在记录完相关路径信息后会进行初始化数据库操作

```python
    # init db
    init_flow_db()
    init_arch_db()
```

### init_flow_db

来源：`from fate_flow.db.db_models import init_database_tables as init_flow_db`

#### init_database_tables 
```python
def init_database_tables():
    members = inspect.getmembers(sys.modules[__name__], inspect.isclass)    # 获取当前模块中的所有类
    table_objs = []
    create_failed_list = []
    for name, obj in members:
        if obj != DataBaseModel and issubclass(obj, DataBaseModel):   # 判断是否是数据库模型类
            table_objs.append(obj)
            LOGGER.info(f"start create table {obj.__name__}")
            try:
                obj.create_table()  # 创建表
                LOGGER.info(f"create table success: {obj.__name__}")
            except Exception as e:
                LOGGER.exception(e)
                create_failed_list.append(obj.__name__)
    if create_failed_list:
        LOGGER.info(f"create tables failed: {create_failed_list}")
        raise Exception(f"create tables failed: {create_failed_list}")
```

DataBaseModel继承至BaseModel，BaseModel继承至peewee.Model，peewee是一个python的ORM框架，用于操作数据库，相关文档在[这里](http://docs.peewee-orm.com/en/latest/peewee/quickstart.html#quickstart)，create_table方法是peewee.Model中的方法，用于创建表，至此创建数据库表的操作就是通过peewee来完成的，已经跟进到fate框架的底层实现了。


相关日志
```bash
cat logs/fate_flow/INFO.log | grep "create table"

[INFO] [2023-08-09 15:59:28,821] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table CacheRecord
[INFO] [2023-08-09 15:59:28,835] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: CacheRecord
[INFO] [2023-08-09 15:59:28,836] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table ComponentInfo
[INFO] [2023-08-09 15:59:28,839] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: ComponentInfo
[INFO] [2023-08-09 15:59:28,839] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table ComponentProviderInfo
[INFO] [2023-08-09 15:59:28,846] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: ComponentProviderInfo
[INFO] [2023-08-09 15:59:28,846] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table ComponentRegistryInfo
[INFO] [2023-08-09 15:59:28,857] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: ComponentRegistryInfo
[INFO] [2023-08-09 15:59:28,857] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table ComponentSummary
[INFO] [2023-08-09 15:59:28,870] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: ComponentSummary
[INFO] [2023-08-09 15:59:28,870] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table DataTableTracking
[INFO] [2023-08-09 15:59:28,875] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: DataTableTracking
[INFO] [2023-08-09 15:59:28,875] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table DependenciesStorageMeta
[INFO] [2023-08-09 15:59:28,880] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: DependenciesStorageMeta
[INFO] [2023-08-09 15:59:28,880] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table EngineRegistry
[INFO] [2023-08-09 15:59:28,891] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: EngineRegistry
[INFO] [2023-08-09 15:59:28,891] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table Job
[INFO] [2023-08-09 15:59:28,901] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: Job
[INFO] [2023-08-09 15:59:28,901] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table MachineLearningModelInfo
[INFO] [2023-08-09 15:59:28,911] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: MachineLearningModelInfo
[INFO] [2023-08-09 15:59:28,911] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table ModelTag
[INFO] [2023-08-09 15:59:28,914] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: ModelTag
[INFO] [2023-08-09 15:59:28,914] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table PipelineComponentMeta
[INFO] [2023-08-09 15:59:28,932] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: PipelineComponentMeta
[INFO] [2023-08-09 15:59:28,932] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table ServerRegistryInfo
[INFO] [2023-08-09 15:59:28,937] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: ServerRegistryInfo
[INFO] [2023-08-09 15:59:28,937] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table ServiceRegistryInfo
[INFO] [2023-08-09 15:59:28,940] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: ServiceRegistryInfo
[INFO] [2023-08-09 15:59:28,940] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table SiteKeyInfo
[INFO] [2023-08-09 15:59:28,947] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: SiteKeyInfo
[INFO] [2023-08-09 15:59:28,947] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table Tag
[INFO] [2023-08-09 15:59:28,952] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: Tag
[INFO] [2023-08-09 15:59:28,952] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table Task
[INFO] [2023-08-09 15:59:28,966] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: Task
[INFO] [2023-08-09 15:59:28,966] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table TrackingMetric
[INFO] [2023-08-09 15:59:28,980] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: TrackingMetric
[INFO] [2023-08-09 15:59:28,980] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table TrackingOutputDataInfo
[INFO] [2023-08-09 15:59:28,992] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: TrackingOutputDataInfo
[INFO] [2023-08-09 15:59:28,992] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:148]: start create table WorkerInfo
[INFO] [2023-08-09 15:59:29,004] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:151]: create table success: WorkerInfo
```

创建了20张表

### init_arch_db

来源：`from fate_arch.metastore.db_models import init_database_tables as init_arch_db`

#### init_database_tables 
```python
def init_database_tables():
    members = inspect.getmembers(sys.modules[__name__], inspect.isclass)    # 获取当前模块中的所有类
    table_objs = []
    create_failed_list = []
    for name, obj in members:
        if obj != DataBaseModel and issubclass(obj, DataBaseModel):   # 判断是否是数据库模型类
            table_objs.append(obj)
            LOGGER.info(f"start create table {obj.__name__}")
            try:
                obj.create_table()  # 创建表
                LOGGER.info(f"create table success: {obj.__name__}")
            except Exception as e:
                LOGGER.exception(e)
                create_failed_list.append(obj.__name__)
    if create_failed_list:
        LOGGER.info(f"create tables failed: {create_failed_list}")
        raise Exception(f"create tables failed: {create_failed_list}")
```

相关日志
```bash
cat logs/fate_flow/INFO.log | grep "create table"

[INFO] [2023-08-09 15:59:29,005] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:85]: start create table SessionRecord
[INFO] [2023-08-09 15:59:29,013] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:88]: create table success: SessionRecord
[INFO] [2023-08-09 15:59:29,013] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:85]: start create table StorageConnectorModel
[INFO] [2023-08-09 15:59:29,018] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:88]: create table success: StorageConnectorModel
[INFO] [2023-08-09 15:59:29,018] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:85]: start create table StorageTableMetaModel
[INFO] [2023-08-09 15:59:29,025] [Server] [2465839:140117571850240] - [db_models.init_database_tables] [line:88]: create table success: StorageTableMetaModel

```

创建了3张表

#### 可知总共有23张表，每张表跟数据库模型类对应


#### *数据库是在哪里创建的？*

发现在$FAET_PROJECT_PATH/fate_flow/db/db_models.py中有

```python
class BaseDataBase:
    def __init__(self):
        database_config = DATABASE.copy()
        db_name = database_config.pop("name")
        if IS_STANDALONE and not bool(int(os.environ.get("FORCE_USE_MYSQL", 0))):
            # sqlite does not support other options
            Insert.on_conflict = lambda self, *args, **kwargs: self.on_conflict_replace()

            from playhouse.apsw_ext import APSWDatabase
            self.database_connection = APSWDatabase(file_utils.get_project_base_directory("fate_sqlite.db"))
            RuntimeConfig.init_config(USE_LOCAL_DATABASE=True)
            stat_logger.info('init sqlite database on standalone mode successfully')
        else:
            self.database_connection = PooledMySQLDatabase(db_name, **database_config)
            stat_logger.info('init mysql database on cluster mode successfully')


DB = BaseDataBase().database_connection
DB.lock = DatabaseLock
```

在单机部署模式下，默认使用sqlite数据库

存放在`$FAET_PROJECT_PATH/fate_sqlite.db`中


#### 查看数据库中的表

<details>

<summary>点击展开</summary>

```bash
(fate_venv) lab@lab-virtual-machine:~/federated_learning/fate/from_src_build/FATE$ sqlite3 fate_sqlite.db
SQLite version 3.37.2 2022-01-06 13:25:41
Enter ".help" for usage hints.
sqlite> .table
componentsummary               t_server_registry_info       
t_cache_record                 t_service_registry_info      
t_component_info               t_session_record             
t_component_provider_info      t_site_key_info              
t_component_registry           t_storage_connector          
t_data_table_tracking          t_storage_table_meta         
t_dependencies_storage_meta    t_tags                       
t_engine_registry              t_task                       
t_job                          t_worker                     
t_machine_learning_model_info  trackingmetric               
t_model_tag                    trackingoutputdatainfo       
t_pipeline_component_meta    
sqlite> 
sqlite> .table
componentsummary               t_server_registry_info       
t_cache_record                 t_service_registry_info      
t_component_info               t_session_record             
t_component_provider_info      t_site_key_info              
t_component_registry           t_storage_connector          
t_data_table_tracking          t_storage_table_meta         
t_dependencies_storage_meta    t_tags                       
t_engine_registry              t_task                       
t_job                          t_worker                     
t_machine_learning_model_info  trackingmetric               
t_model_tag                    trackingoutputdatainfo       
t_pipeline_component_meta    
sqlite> PRAGMA table_info(t_component_info);
0|f_component_name|VARCHAR(30)|1||1
1|f_create_time|INTEGER|0||0
2|f_create_date|DATETIME|0||0
3|f_update_time|INTEGER|0||0
4|f_update_date|DATETIME|0||0
5|f_component_alias|LONGTEXT|1||0
6|f_default_provider|VARCHAR(20)|1||0
7|f_support_provider|LONGTEXT|0||0
sqlite> SELECT * FROM t_component_info;
download|1691567969487|2023-08-09 15:59:29|1691567969891|2023-08-09 15:59:29|["Download"]|fate_flow|["fate_flow"]
Download|1691567969489|2023-08-09 15:59:29|1691567969893|2023-08-09 15:59:29|["Download"]|fate_flow|["fate_flow"]
reader|1691567969491|2023-08-09 15:59:29|1691567969895|2023-08-09 15:59:29|["Reader"]|fate_flow|["fate_flow"]
Reader|1691567969494|2023-08-09 15:59:29|1691567969897|2023-08-09 15:59:29|["Reader"]|fate_flow|["fate_flow"]
modelloader|1691567969496|2023-08-09 15:59:29|1691567969899|2023-08-09 15:59:29|["ModelLoader"]|fate_flow|["fate_flow"]
ModelLoader|1691567969498|2023-08-09 15:59:29|1691567969902|2023-08-09 15:59:29|["ModelLoader"]|fate_flow|["fate_flow"]
apireader|1691567969500|2023-08-09 15:59:29|1691567969904|2023-08-09 15:59:29|["ApiReader"]|fate_flow|["fate_flow"]
ApiReader|1691567969502|2023-08-09 15:59:29|1691567969906|2023-08-09 15:59:29|["ApiReader"]|fate_flow|["fate_flow"]
modelrestore|1691567969505|2023-08-09 15:59:29|1691567969908|2023-08-09 15:59:29|["ModelRestore"]|fate_flow|["fate_flow"]
ModelRestore|1691567969507|2023-08-09 15:59:29|1691567969910|2023-08-09 15:59:29|["ModelRestore"]|fate_flow|["fate_flow"]
modelstore|1691567969510|2023-08-09 15:59:29|1691567969912|2023-08-09 15:59:29|["ModelStore"]|fate_flow|["fate_flow"]
ModelStore|1691567969512|2023-08-09 15:59:29|1691567969914|2023-08-09 15:59:29|["ModelStore"]|fate_flow|["fate_flow"]
upload|1691567969514|2023-08-09 15:59:29|1691567969916|2023-08-09 15:59:29|["Upload"]|fate_flow|["fate_flow"]
Upload|1691567969516|2023-08-09 15:59:29|1691567969918|2023-08-09 15:59:29|["Upload"]|fate_flow|["fate_flow"]
cacheloader|1691567969519|2023-08-09 15:59:29|1691567969920|2023-08-09 15:59:29|["CacheLoader"]|fate_flow|["fate_flow"]
CacheLoader|1691567969521|2023-08-09 15:59:29|1691567969923|2023-08-09 15:59:29|["CacheLoader"]|fate_flow|["fate_flow"]
writer|1691567969523|2023-08-09 15:59:29|1691567969925|2023-08-09 15:59:29|["Writer"]|fate_flow|["fate_flow"]
Writer|1691567969526|2023-08-09 15:59:29|1691567969927|2023-08-09 15:59:29|["Writer"]|fate_flow|["fate_flow"]
datatransform|1691567969929|2023-08-09 15:59:29|1691567969929|2023-08-09 15:59:29|["DataTransform"]|fate|["fate"]
DataTransform|1691567969932|2023-08-09 15:59:29|1691567969932|2023-08-09 15:59:29|["DataTransform"]|fate|["fate"]
ftl|1691567969934|2023-08-09 15:59:29|1691567969934|2023-08-09 15:59:29|["FTL"]|fate|["fate"]
FTL|1691567969936|2023-08-09 15:59:29|1691567969936|2023-08-09 15:59:29|["FTL"]|fate|["fate"]
custnn|1691567969939|2023-08-09 15:59:29|1691567969939|2023-08-09 15:59:29|["CustNN"]|fate|["fate"]
CustNN|1691567969941|2023-08-09 15:59:29|1691567969941|2023-08-09 15:59:29|["CustNN"]|fate|["fate"]
feldmanverifiablesum|1691567969943|2023-08-09 15:59:29|1691567969943|2023-08-09 15:59:29|["FeldmanVerifiableSum"]|fate|["fate"]
FeldmanVerifiableSum|1691567969946|2023-08-09 15:59:29|1691567969946|2023-08-09 15:59:29|["FeldmanVerifiableSum"]|fate|["fate"]
localbaseline|1691567969948|2023-08-09 15:59:29|1691567969948|2023-08-09 15:59:29|["LocalBaseline"]|fate|["fate"]
LocalBaseline|1691567969950|2023-08-09 15:59:29|1691567969950|2023-08-09 15:59:29|["LocalBaseline"]|fate|["fate"]
heterosecureboost|1691567969952|2023-08-09 15:59:29|1691567969952|2023-08-09 15:59:29|["HeteroSecureBoost"]|fate|["fate"]
HeteroSecureBoost|1691567969955|2023-08-09 15:59:29|1691567969955|2023-08-09 15:59:29|["HeteroSecureBoost"]|fate|["fate"]
scorecard|1691567969958|2023-08-09 15:59:29|1691567969958|2023-08-09 15:59:29|["Scorecard"]|fate|["fate"]
Scorecard|1691567969960|2023-08-09 15:59:29|1691567969960|2023-08-09 15:59:29|["Scorecard"]|fate|["fate"]
heterolr|1691567969963|2023-08-09 15:59:29|1691567969963|2023-08-09 15:59:29|["HeteroLR"]|fate|["fate"]
HeteroLR|1691567969965|2023-08-09 15:59:29|1691567969965|2023-08-09 15:59:29|["HeteroLR"]|fate|["fate"]
heterofastsecureboost|1691567969967|2023-08-09 15:59:29|1691567969967|2023-08-09 15:59:29|["HeteroFastSecureBoost"]|fate|["fate"]
HeteroFastSecureBoost|1691567969970|2023-08-09 15:59:29|1691567969970|2023-08-09 15:59:29|["HeteroFastSecureBoost"]|fate|["fate"]
onehotencoder|1691567969972|2023-08-09 15:59:29|1691567969972|2023-08-09 15:59:29|["OneHotEncoder"]|fate|["fate"]
OneHotEncoder|1691567969974|2023-08-09 15:59:29|1691567969974|2023-08-09 15:59:29|["OneHotEncoder"]|fate|["fate"]
featurescale|1691567969977|2023-08-09 15:59:29|1691567969977|2023-08-09 15:59:29|["FeatureScale"]|fate|["fate"]
FeatureScale|1691567969979|2023-08-09 15:59:29|1691567969979|2023-08-09 15:59:29|["FeatureScale"]|fate|["fate"]
homosecureboost|1691567969982|2023-08-09 15:59:29|1691567969982|2023-08-09 15:59:29|["HomoSecureBoost", "HomoSecureboost"]|fate|["fate"]
HomoSecureBoost|1691567969984|2023-08-09 15:59:29|1691567969984|2023-08-09 15:59:29|["HomoSecureBoost", "HomoSecureboost"]|fate|["fate"]
HomoSecureboost|1691567969986|2023-08-09 15:59:29|1691567969986|2023-08-09 15:59:29|["HomoSecureBoost", "HomoSecureboost"]|fate|["fate"]
heterofeatureselection|1691567969989|2023-08-09 15:59:29|1691567969989|2023-08-09 15:59:29|["HeteroFeatureSelection"]|fate|["fate"]
HeteroFeatureSelection|1691567969991|2023-08-09 15:59:29|1691567969991|2023-08-09 15:59:29|["HeteroFeatureSelection"]|fate|["fate"]
columnexpand|1691567969993|2023-08-09 15:59:29|1691567969993|2023-08-09 15:59:29|["ColumnExpand"]|fate|["fate"]
ColumnExpand|1691567969996|2023-08-09 15:59:29|1691567969996|2023-08-09 15:59:29|["ColumnExpand"]|fate|["fate"]
heterosshelinr|1691567969998|2023-08-09 15:59:29|1691567969998|2023-08-09 15:59:29|["HeteroSSHELinR"]|fate|["fate"]
HeteroSSHELinR|1691567970001|2023-08-09 15:59:30|1691567970001|2023-08-09 15:59:30|["HeteroSSHELinR"]|fate|["fate"]
homofeaturebinning|1691567970003|2023-08-09 15:59:30|1691567970003|2023-08-09 15:59:30|["HomoFeatureBinning"]|fate|["fate"]
HomoFeatureBinning|1691567970005|2023-08-09 15:59:30|1691567970005|2023-08-09 15:59:30|["HomoFeatureBinning"]|fate|["fate"]
heteropoisson|1691567970007|2023-08-09 15:59:30|1691567970007|2023-08-09 15:59:30|["HeteroPoisson"]|fate|["fate"]
HeteroPoisson|1691567970010|2023-08-09 15:59:30|1691567970010|2023-08-09 15:59:30|["HeteroPoisson"]|fate|["fate"]
heterosshelr|1691567970012|2023-08-09 15:59:30|1691567970012|2023-08-09 15:59:30|["HeteroSSHELR"]|fate|["fate"]
HeteroSSHELR|1691567970015|2023-08-09 15:59:30|1691567970015|2023-08-09 15:59:30|["HeteroSSHELR"]|fate|["fate"]
heteronn|1691567970018|2023-08-09 15:59:30|1691567970018|2023-08-09 15:59:30|["HeteroNN"]|fate|["fate"]
HeteroNN|1691567970020|2023-08-09 15:59:30|1691567970020|2023-08-09 15:59:30|["HeteroNN"]|fate|["fate"]
secureaddexample|1691567970022|2023-08-09 15:59:30|1691567970022|2023-08-09 15:59:30|["SecureAddExample"]|fate|["fate"]
SecureAddExample|1691567970024|2023-08-09 15:59:30|1691567970024|2023-08-09 15:59:30|["SecureAddExample"]|fate|["fate"]
heteropearson|1691567970026|2023-08-09 15:59:30|1691567970026|2023-08-09 15:59:30|["HeteroPearson"]|fate|["fate"]
HeteroPearson|1691567970029|2023-08-09 15:59:30|1691567970029|2023-08-09 15:59:30|["HeteroPearson"]|fate|["fate"]
featureimputation|1691567970031|2023-08-09 15:59:30|1691567970031|2023-08-09 15:59:30|["FeatureImputation"]|fate|["fate"]
FeatureImputation|1691567970033|2023-08-09 15:59:30|1691567970034|2023-08-09 15:59:30|["FeatureImputation"]|fate|["fate"]
homoonehotencoder|1691567970036|2023-08-09 15:59:30|1691567970036|2023-08-09 15:59:30|["HomoOneHotEncoder"]|fate|["fate"]
HomoOneHotEncoder|1691567970038|2023-08-09 15:59:30|1691567970038|2023-08-09 15:59:30|["HomoOneHotEncoder"]|fate|["fate"]
psi|1691567970041|2023-08-09 15:59:30|1691567970041|2023-08-09 15:59:30|["PSI"]|fate|["fate"]
PSI|1691567970043|2023-08-09 15:59:30|1691567970043|2023-08-09 15:59:30|["PSI"]|fate|["fate"]
heterodatasplit|1691567970045|2023-08-09 15:59:30|1691567970045|2023-08-09 15:59:30|["HeteroDataSplit"]|fate|["fate"]
HeteroDataSplit|1691567970047|2023-08-09 15:59:30|1691567970048|2023-08-09 15:59:30|["HeteroDataSplit"]|fate|["fate"]
secureinformationretrieval|1691567970050|2023-08-09 15:59:30|1691567970050|2023-08-09 15:59:30|["SecureInformationRetrieval"]|fate|["fate"]
SecureInformationRetrieval|1691567970052|2023-08-09 15:59:30|1691567970052|2023-08-09 15:59:30|["SecureInformationRetrieval"]|fate|["fate"]
federatedsample|1691567970054|2023-08-09 15:59:30|1691567970054|2023-08-09 15:59:30|["FederatedSample"]|fate|["fate"]
FederatedSample|1691567970056|2023-08-09 15:59:30|1691567970056|2023-08-09 15:59:30|["FederatedSample"]|fate|["fate"]
sampleweight|1691567970059|2023-08-09 15:59:30|1691567970059|2023-08-09 15:59:30|["SampleWeight"]|fate|["fate"]
SampleWeight|1691567970061|2023-08-09 15:59:30|1691567970061|2023-08-09 15:59:30|["SampleWeight"]|fate|["fate"]
dataio|1691567970063|2023-08-09 15:59:30|1691567970063|2023-08-09 15:59:30|["DataIO"]|fate|["fate"]
DataIO|1691567970066|2023-08-09 15:59:30|1691567970066|2023-08-09 15:59:30|["DataIO"]|fate|["fate"]
heterofeaturebinning|1691567970068|2023-08-09 15:59:30|1691567970068|2023-08-09 15:59:30|["HeteroFeatureBinning"]|fate|["fate"]
HeteroFeatureBinning|1691567970071|2023-08-09 15:59:30|1691567970071|2023-08-09 15:59:30|["HeteroFeatureBinning"]|fate|["fate"]
homodatasplit|1691567970073|2023-08-09 15:59:30|1691567970073|2023-08-09 15:59:30|["HomoDataSplit"]|fate|["fate"]
HomoDataSplit|1691567970075|2023-08-09 15:59:30|1691567970075|2023-08-09 15:59:30|["HomoDataSplit"]|fate|["fate"]
intersection|1691567970077|2023-08-09 15:59:30|1691567970077|2023-08-09 15:59:30|["Intersection"]|fate|["fate"]
Intersection|1691567970080|2023-08-09 15:59:30|1691567970080|2023-08-09 15:59:30|["Intersection"]|fate|["fate"]
union|1691567970083|2023-08-09 15:59:30|1691567970083|2023-08-09 15:59:30|["Union"]|fate|["fate"]
Union|1691567970085|2023-08-09 15:59:30|1691567970085|2023-08-09 15:59:30|["Union"]|fate|["fate"]
spdztest|1691567970087|2023-08-09 15:59:30|1691567970087|2023-08-09 15:59:30|["SPDZTest"]|fate|["fate"]
SPDZTest|1691567970089|2023-08-09 15:59:30|1691567970089|2023-08-09 15:59:30|["SPDZTest"]|fate|["fate"]
labeltransform|1691567970091|2023-08-09 15:59:30|1691567970091|2023-08-09 15:59:30|["LabelTransform"]|fate|["fate"]
LabelTransform|1691567970094|2023-08-09 15:59:30|1691567970094|2023-08-09 15:59:30|["LabelTransform"]|fate|["fate"]
homolr|1691567970096|2023-08-09 15:59:30|1691567970096|2023-08-09 15:59:30|["HomoLR"]|fate|["fate"]
HomoLR|1691567970098|2023-08-09 15:59:30|1691567970098|2023-08-09 15:59:30|["HomoLR"]|fate|["fate"]
heterokmeans|1691567970100|2023-08-09 15:59:30|1691567970100|2023-08-09 15:59:30|["HeteroKmeans"]|fate|["fate"]
HeteroKmeans|1691567970103|2023-08-09 15:59:30|1691567970103|2023-08-09 15:59:30|["HeteroKmeans"]|fate|["fate"]
heterolinr|1691567970105|2023-08-09 15:59:30|1691567970105|2023-08-09 15:59:30|["HeteroLinR"]|fate|["fate"]
HeteroLinR|1691567970108|2023-08-09 15:59:30|1691567970108|2023-08-09 15:59:30|["HeteroLinR"]|fate|["fate"]
evaluation|1691567970110|2023-08-09 15:59:30|1691567970110|2023-08-09 15:59:30|["Evaluation"]|fate|["fate"]
Evaluation|1691567970112|2023-08-09 15:59:30|1691567970112|2023-08-09 15:59:30|["Evaluation"]|fate|["fate"]
datastatistics|1691567970114|2023-08-09 15:59:30|1691567970114|2023-08-09 15:59:30|["DataStatistics"]|fate|["fate"]
DataStatistics|1691567970117|2023-08-09 15:59:30|1691567970117|2023-08-09 15:59:30|["DataStatistics"]|fate|["fate"]
positiveunlabeled|1691567970119|2023-08-09 15:59:30|1691567970119|2023-08-09 15:59:30|["PositiveUnlabeled"]|fate|["fate"]
PositiveUnlabeled|1691567970121|2023-08-09 15:59:30|1691567970121|2023-08-09 15:59:30|["PositiveUnlabeled"]|fate|["fate"]
homonn|1691567970124|2023-08-09 15:59:30|1691567970124|2023-08-09 15:59:30|["HomoNN"]|fate|["fate"]
HomoNN|1691567970128|2023-08-09 15:59:30|1691567970128|2023-08-09 15:59:30|["HomoNN"]|fate|["fate"]
sqlite> 
```
</detail>



#### 对比$FAET_PROJECT_PATH/fate_flow/db/db_models.py中的ComponentInfo类

```python
class ComponentInfo(DataBaseModel):
    f_component_name = CharField(max_length=30, primary_key=True)
    f_component_alias = JSONField()
    f_default_provider = CharField(max_length=20)
    f_support_provider = ListField(null=True)

    class Meta:
        db_table = "t_component_info"

```

可知，确实有23张表，每张表跟数据库模型类对应。



## 数据库操作，调用分析

<!-- TODO -->