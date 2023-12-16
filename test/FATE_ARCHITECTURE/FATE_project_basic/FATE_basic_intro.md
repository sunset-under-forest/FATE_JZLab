# FATE框架开发介绍

## 前置知识准备

- 相关环境变量
    - `FATE_PROJECT_BASE`：当前的fate项目根目录
    - `PYTHONPATH`：供当前的python查找包目录的路径
- 联邦学习基础知识介绍
- 目录结构介绍

```
FATE_PROJECT_BASE
├── bin (命令行工具，用于切换python环境，配置环境变量等)
├── c   (代理相关)
├── conf    (配置文件，FATE基本配置选项都从这里读取，重要)
├── deploy  (部署相关，就是一些教程，不重要)
├── doc (文档)
├── eggroll (另外的仓库，集群相关的，暂时没研究)
├── examples    (示例代码，包括示例所需的数据集)
├── fateboard   (FATE的web管理界面，就是个web服务，跟算法逻辑没关系)
├── fateflow    (FATE flow server，FATE的核心，负责算法任务的调度，任务状态管理，算法组件的注册和调用等，FATE的心脏)
├── python  (FATE的python SDK，包括算法组件的实现，算法任务的提交，数据管理等，可以说是FATE的大脑，开发算法最重要的部分)
├── rust    (一些rust代码，暂时没研究，主要用于加密，可能是未来FATE会加入的功能)
```

python目录树

``````
python/
├── fate_arch   
├── fate_client   
├── fate_test
├── federatedml

``````

- fate_arch
    (FATE架构包，包含很多FATE的基类，常量，配置信息，基础对象结构等)
- fate_client
    (FATE客户端包，包含FATE的客户端SDK，部署的时候需要pip 安装，自带命令行工具，主要的用户接口都在这里，只包含任务提交，作业，不包含算法组件的实现) 
- fate_test
    (FATE测试包，包含FATE的单元测试，可以参考使用)
- federatedml
    (联邦学习算法包，FATE框架最重要的部分，FATE的所有算法组件，支持的模型，以及联邦学习的各种协议，算法组件的实现都在这里，是我们研究的重点)

## 测试组件编写

> 目标：编写一个FATE框架可调用的组件，组件名为ATest

1. 首先声明相关环境变量

    算法目录环境变量：

    - 如果是**源码部署**`export FEDERATED_ML_PATH=$FATE_PROJECT_BASE/python/federatedml/`

    - 如果是**docker**部署`export FEDERATED_ML_PATH=$FATE_PROJECT_BASE/fate/python/federatedml/`

    ```bash
    export TEST_COMPONENT_NAME=ATest
    
    # 进入算法包目录下
    cd $TEST_COMPONENT_NAME
    ```

    

2. 进入param文件夹并创建`a_test_param.py`文件，内部包含组件参数类

    ```bash
    vim param/a_test_param.py
    ```

    `a_test_param.py`文件内容

    ```python
    from federatedml.param.base_param import BaseParam
    from federatedml.util import LOGGER
    
    
    class ATestParam(BaseParam):
        """
        TEST
    
        Parameters
        ----------
        param1 : None or int, default: None
            Specify the random state for shuffle.
        param2 : float or int or None, default: 0.0
            Specify test data set size.
            
        """
    
        def __init__(self, param1=None, param2=None):
            super(ATestParam, self).__init__()       # super() 函数是用于调用父类(超类)的一个方法。
            self.param1 = param1
            self.param2 = param2
    
        def check(self):
            model_param_descr = "a test param's "
            if self.param1 is not None:
                if not isinstance(self.param1, int):
                    raise ValueError(f"{model_param_descr} param1 should be int type")
                BaseParam.check_nonnegative_number(self.param1, f"{model_param_descr} param1 ")
    
            if self.param2 is not None:
                BaseParam.check_nonnegative_number(self.param2, f"{model_param_descr} param2 ")
                if isinstance(self.param2, float):
                    BaseParam.check_decimal_float(self.param2, f"{model_param_descr} param2 ")
    
    ```

    

3. 创建组件实体目录包并编写`a_test.py`文件

```bash
mkdir -p $TEST_COMPONENT_NAME
vim $TEST_COMPONENT_NAME/a_test.py
```

`a_test.py`文件内容，包含组件的具体实现

```python
import sys
from federatedml.model_base import ModelBase
from federatedml.param.a_test_param import ATestParam
import ctypes
import requests


class ATest(ModelBase):
    def __init__(self):
        super().__init__()
        self.model_param = ATestParam()
        self.model_name = 'ATest'

    def fit(self, *args):
        """
        测试
        """
        port = "8000"
        url = "http://127.0.0.1:" + port + "/"
        response = requests.get(url + "ATest has been called")
        return response.text


```

4. 在组件包下创建`a_test.py`组件接口文件

```bash
vim components/a_test.py
```

`components/a_test.py`文件内容，要符合FATE框架组件开发规范

```python
from .components import ComponentMeta

a_test_cpn_meta = ComponentMeta("ATest")

@a_test_cpn_meta.bind_param
def a_test_param():
    from federatedml.param.a_test_param import ATestParam

    return ATestParam

@a_test_cpn_meta.bind_runner.on_guest.on_local.on_host.on_arbiter
def a_test_runner():
    from federatedml.ATest.a_test import ATest
    return ATest


```

5. 重启fateflow

```bash
cd $FATE_PROJECT_BASE/fateflow
bash bin/service.sh restart
```

6. 输入命令查看组件是否注册成功

```bash
flow provider list |grep "atest"
```

7. 准备进行组件测试

```bash
# 创建测试文件夹
cd $FATE_PROJECT_BASE && mkdir -p my_test && cd my_test 

# 准备测试数据
cp $FATE_PROJECT_BASE/examples/data/breast_homo_guest.csv test_data.csv

# 编写数据上传任务配置文件
vim upload_conf.json
```

`upload_conf.json`内容

```json
{
    "file": "{FATE_PROJECT_BASH}/my_test/test_data.csv",
    "table_name": "test_data",
    "namespace": "test",
    "head": 1,
    "partition": 8
}
```

```bash
# 上传数据
flow data upload -c upload_conf.json

# 编写任务配置文件
vim test_conf.json
vim test_dsl.json
```

`test_conf.json`内容

```json
{
    "dsl_version": 2,
    "initiator": {
        "role": "guest",
        "party_id": 9999
    },
    "role": {
        "guest": [
            9999
        ]
    },
    "component_parameters": {
        "common": {
            "ATest": {
                    "param1": 7,
		    "param2": 0.1
            }
        },
        "role": {
            "guest": {
                "0": {
                    "reader_0": {
                        "table": {
                            "name": "test_data",
                            "namespace": "test"
                        }
                    },
                    "data_transform_0": {
                        "with_label": true,
                        "output_format": "dense"
                    },
                    "data_transform_1": {
                        "with_label": true,
                        "output_format": "dense"
                    }
                }
            }
	}
    }
}
```

`test_dsl.json`内容

```json
{
    "components": {
        "reader_0": {
            "module": "Reader",
            "output": {
                "data": [
                    "data"
                ]
            }
        },
        "data_transform_0": {
            "module": "DataTransform",
            "input": {
                "data": {
                    "data": [
                        "reader_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ],
                "model": [
                    "model"
                ]
            }
        },
        "a_test_0": {
            "module": "ATest",
            "input": {
                "data": {
                    "train_data": [
                        "data_transform_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ],
                "model": [
                    "model"
                ]
            }
        }
    }
}
```

7. 执行任务

```bash
flow job submit -c test_conf.json -d test_dsl.json
```

## 对密码类库的调用

目前大多数项目都涵盖了python和c++混合编程的需求，在FATE框架中使用一些密码类库也是一样，需要做好python语言和c++语言之间的适配，要明确我们的主要目标是在python中使用c++的库，目前我探索到的，可行的方案有：

| 方案                                                        | 优点                                                   | 缺点                                                         | 开发时间          |
| ----------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------ | ----------------- |
| 直接用python重写c++类库                                     | 后续编程直接用python写，方便，还能加深对相关库的理解。 | 重写过程的学习成本很大，要花费大量的时间代码重构以及调试，而且后续的库函数性能由于python的原因会很差。 | 最长              |
| 使用动态链接库接口封装c++代码，通过python调用               | 保持了python部分编程的便利的同时还兼具c++的效率。      | 只能导出少数的已有的库函数，如果项目需要基于库函数进行二次开发，需要足够的c++编程基础，以及一定的开发时间。 | 适中（有c++基础） |
| 通过网络协议进行端口路由编程，类似rpc协议远程函数调用的形式 | 跨平台性好，使用方便简洁，适合大项目运转。             | 需要将类库重构为内核，以及规定相关函数的接口规范，还需要网络编程知识，甚至是自己实现底层协议，有一定的开发周期。 | 长                |

