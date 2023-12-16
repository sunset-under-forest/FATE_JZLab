# Pipeline Python API

Pipeline跟flow的功能差不多，也有命令行工具（仅用于建立和fate flow server 的连接），但同时也支持python API（建立连接后跟fate flow server进行交互），实际上是一个python包，可以用来进行开发，相当于一个SDK，可编程的遥控器，涉及到具体的业务逻辑。（开发型）

## 通过上传数据例子进行分析

##### 上传数据

```python
from pipeline.backend.pipeline import PipeLine

# 初始化Pipeline实例
pipeline_upload = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999)

# 选择储存数据分区
partition = 4

# 数据的配置信息
dense_data_guest = {"name": "breast_hetero_guest", "namespace": f"experiment"}
dense_data_host = {"name": "breast_hetero_host", "namespace": f"experiment"}
tag_data = {"name": "breast_hetero_host", "namespace": f"experiment"}


import os

# 数据的路径
data_base = "/home/lab/federated_learning/fate/from_src_build/FATE/"
pipeline_upload.

# 添加上传数据的任务
add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_guest.csv"),
                                table_name=dense_data_guest["name"],             # table name
                                namespace=dense_data_guest["namespace"],         # namespace
                                head=1, partition=partition)               # data info

pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_host.csv"),
                                table_name=dense_data_host["name"],
                                namespace=dense_data_host["namespace"],
                                head=1, partition=partition)

pipeline_upload.add_upload_data(file=os.path.join(data_base, "examples/data/breast_hetero_host.csv"),
                                table_name=tag_data["name"],
                                namespace=tag_data["namespace"],
                                head=1, partition=partition)

# 开始上传数据
pipeline_upload.upload(drop=1)
```


### 分析调用链

```
1. 首先定位到Pipeline类的add_upload_data方法
该方法的作用时添加上传数据的任务，将传入的相关参数进行解析，并将解析后的参数存储在成员变量_upload_conf中，成员变量_upload_conf是一个列表，列表中的每一个元素都是一个字典。

2. 然后定位到Pipeline类的upload方法
该方法的核心上传部分是
self._train_job_id, detail_info = self._job_invoker.upload_data(upload_conf, int(drop))
这一句将相对应的参数通过调用self._job_invoker的upload_data方法进行上传。

3. 根据self._job_invoker的类型，发现self._job_invoker是pipeline.utils.invoker.job_submitter中JobInvoker类的一个实例

4. 定位到JobInvoker类
self.client = FlowClient(ip=conf.PipelineConfig.IP, port=conf.PipelineConfig.PORT, version=conf.SERVER_VERSION,
                                 app_key=conf.PipelineConfig.APP_KEY, secret_key=conf.PipelineConfig.SECRET_KEY)
在创建对象时JobInvoker会初始化一个flow_sdk.client中的FlowClient对象，并传入配置好的对应的FATE Flow Server的ip和port。

在该类的upload_data方法中，会调用FlowClient的data成员的upload接口，进行数据上传。
result = self.client.data.upload(config_data=submit_conf, verbose=1, drop=drop)

5. 定位到FlowClient类
FlowCient是flow_sdk.client.base中BaseFlowClient的子类，data是这个类的静态成员，是flow_sdk.client.api包下Data类的一个实例。

6. 定位到Data类
Data是flow_sdk.client.api.base中BaseFlowAPI的子类，Data实现的upload方法中
return self._post(url='data/upload', data=data,
                                      params=json_dumps(config_data), headers={'Content-Type': data.content_type})
这是上传数据的主要代码，根据分析可知use_local_data的字段是判断使用本地还是Fate Flow Server的数据，默认为True，即使用本地数据并上传至Fate Flow Server。

7. 定位到BaseFlowAPI类
发现这是一个抽象类，虽然有_post方法，但是_post方法中上传部分的中心代码
self._client.post(url, **kwargs)
又调用了成员变量的_client的post方法，但是_client是一个在__init__方法中传入的参数，初始为None，Data类也没有实现自己的__init__，而且前面在实例化Data的时候是没有传入参数的
data = api.Data()
所以可见这个类是抽象的，只是一个接口，具体的实现，或者说_client的赋值一定在别的地方

8. 返回到FlowClient类
可以发现在FlowClient中的__init__方法中，会调用父类BaseFlowClient的__init__方法。

9. 定位到BaseFlowClient类
该类也实现了post方法，这里就可以直接追溯到底层了，是用requests库实现的发送post请求，但是关键实现调用的代码是这一部分


def _is_api_endpoint(obj):
    return isinstance(obj, BaseFlowAPI)

class BaseFlowClient:
    API_BASE_URL = ''

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        api_endpoints = inspect.getmembers(self, _is_api_endpoint)
        for name, api in api_endpoints:
            api_cls = type(api)
            api = api_cls(self)
            setattr(self, name, api)
        return self

这部分代码实现了一个动态绑定的功能，因为前面FlowClient类的data成员变量是静态成员，所以在实例化FlowClient之前，data成员变量就已经初始化了，于是在实例化FlowClient的时候，通过__new__函数中的
api_endpoints = inspect.getmembers(self, _is_api_endpoint)
就可以知道当前对象拥有的所有成员变量并筛选出其中的api成员变量（继承BaseFlowAPI的类实例），通过
api = api_cls(self)
将每个api成员变量重新初始化，在此时就实现了将BaseFlowAPI的_client成员变量赋值为当前BaseFlowClient对象的功能，妙哉。
```

