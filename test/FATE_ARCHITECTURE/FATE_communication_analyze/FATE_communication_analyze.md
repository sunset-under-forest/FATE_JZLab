# FATE框架底层通信分析

FATE框架的通信层功能实现主要在`fate_arch.federation`包下

## 垃圾回收部分
为什么首先是垃圾回收部分呢，因为首先看的`fate_arch.federation.transfer_variable`中，存在唯一涉及逻辑的非底层（非标准库）导入就是`fate_arch.federation._gc`中的IterationGC。

以下是相关的文件

### fate_arch/abc/_gc.py
垃圾回收接口，定义了一个抽象方法`add_gc_action`

### fate_arch/federation/_gc.py

`IterationGC`实现了一个简单的垃圾回收框架，通过迭代的方式进行垃圾回收，同时提供了禁用、设置容量等方法。

AI模拟的使用方法

```PYTHON
# 创建一个 IterationGC 实例
gc_manager = IterationGC()

# 添加垃圾回收动作
gc_manager.add_gc_action("tag1", obj1, "cleanup_method", {"param": "value"})
gc_manager.add_gc_action("tag1", obj2, "cleanup_method", {"param": "value"})
gc_manager.add_gc_action("tag2", obj3, "cleanup_method", {"param": "value"})

# 执行垃圾回收
gc_manager.gc()

# 禁用垃圾回收
gc_manager.disable()

# 设置垃圾回收轮容量
gc_manager.set_capacity(3)

# 再次添加垃圾回收动作
gc_manager.add_gc_action("tag3", obj4, "cleanup_method", {"param": "value"})
gc_manager.add_gc_action("tag3", obj5, "cleanup_method", {"param": "value"})
gc_manager.add_gc_action("tag3", obj6, "cleanup_method", {"param": "value"})

# 执行垃圾回收
gc_manager.gc()

# 清理所有的垃圾回收
gc_manager.clean()

```

疑惑：如果对于`add_gc_action`来说是通过`method`来删除或回收`obj`，那么`method`是`obj`的成员方法的话，析构函数就可以满足这个场景，所以考虑到目的并不一定是删除`obj`这个对象，也有可能是通过这个对象去实现对其他对象的管理，类似`管理器对象`（`Manager`）

**可以抽象出`IterationGC`是一个垃圾回收器，类似垃圾桶，用于清理回收对象，`add_gc_action`方法添加相应的对象以及垃圾回收方法（字符串形式）及参数，调用`gc`方法整理自身，如果带回收的对象已满，则回收队列第一个，`clean`则会回收队列中所有的对象。**

类图：

![gc.svg](images%2Fgc.svg)

<HR>

## 通信变量部分

### fate_arch/federation/transfer_variable.py

#### FederationTagNamespace

`FederationTagNamespace`是一个命名管理器或标签生成器。

![FederationTagNamespace.svg](images%2FFederationTagNamespace.svg)

- `set_namespace` 方法允许在运行时更改命名空间，这样在一些特定情境下，可以动态地修改标签的命名空间。
- `generate_tag` 方法用于生成标签。它将命名空间和所有后缀参数连接在一起，形成一个点分隔的字符串作为标签。这是一个常见的做法，用于在分布式系统中生成全局唯一的标识符。

> 设计模式：单例模式





#### Variable

![Variable.svg](images%2FVariable.svg)


`Variable`这个类的主要作用是管理变量的信息，包括名称、源方和目标方的角色，以及进行本地和远程的垃圾回收。其方法主要用于设置变量属性、进行对象的发送和接收，并提供一些控制垃圾回收的功能。简单点说就是通信（联邦）过程中对需要传输的对象的包装，这个变量只是一个名字（对传输过程的抽象），需要在各个参与方中定义，通过`remote`发送对象，`get`获取对象。

通过底层的分析，发送过程首先会根据传输过程变量的`name`属性生成唯一标识符，存储在数据库中的元数据表（所有party都可以访问）中，然后将序列化后的传输对象存储也存入数据库（所有party都可以访问）中，写入元数据表和存储真实对象过程是加锁的，防止异步分布式访问冲突问题，接收变量过程则会先查询元数据表，如果没有查询到则异步循环等待，查询到后就会相应地将数据库中的对象反序列化后返回，整个过程同样要防止访问冲突问题，这样就实现了传输过程中对传输变量的抽象。

#### BaseTransferVariables

![BaseTransferVariables.svg](images%2FBaseTransferVariables.svg)

定义的传输变量的基类，开发者定义自己的传输变量要继承的类，实现了`_create_variable`接口，或者说自定义的传输类（或者需要传输的数据结构）通过`_create_variable`方法声明传输变量。

整体的传输过程如下：
![Variable_transfer_process.svg](images%2FVariable_transfer_process.svg)


> 使用示例也可如下图

#### federatedml/transfer_variable/transfer_class/secret_share_transfer_variable.py

```PYTHON
class SecretShareTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.multiply_triplets_cross = self._create_variable(
            name='multiply_triplets_cross', src=[
                'guest', 'host'], dst=[
                'guest', 'host'])
        self.multiply_triplets_encrypted = self._create_variable(
            name='multiply_triplets_encrypted', src=[
                'guest', 'host'], dst=[
                'guest', 'host'])
        self.rescontruct = self._create_variable(name='rescontruct', src=['guest', 'host'], dst=['guest', 'host'])
        self.share = self._create_variable(name='share', src=['guest', 'host'], dst=['guest', 'host'])
        self.encrypted_share_matrix = self._create_variable(
            name='encrypted_share_matrix', src=[
                'guest', "host"], dst=[
                'host', "guest"])
        self.q_field = self._create_variable(
            name='q_field', src=[
                'guest', "host"], dst=[
                'host', "guest"])
```

通过继承`BaseTransferVariables`实现的`SecretShareTransferVariable`类，作为秘密分享传输变量（对象），定义数据结构组成如下：

| 变量名                      |                  |
| --------------------------- | ---------------- |
| multiply_triplets_cross     | 乘法三元组       |
| multiply_triplets_encrypted | 加密的乘法三元组 |
| rescontruct                 | 重组变量         |
| share                       | 份额             |
| encrypted_share_matrix      | 加密的份额矩阵   |
| q_field                     | 有限域           |

### federatedml/secureprotol/spdz/communicator/federation.py

```PYTHON
class Communicator(object):

    def __init__(self, local_party=None, all_parties=None):
        self._transfer_variable = SecretShareTransferVariable()
        self._share_variable = self._transfer_variable.share.disable_auto_clean()
        self._rescontruct_variable = self._transfer_variable.rescontruct.set_preserve_num(3)
        self._mul_triplets_encrypted_variable = self._transfer_variable.multiply_triplets_encrypted.set_preserve_num(3)
        self._mul_triplets_cross_variable = self._transfer_variable.multiply_triplets_cross.set_preserve_num(3)
        self._q_field_variable = self._transfer_variable.q_field.disable_auto_clean()

        self._local_party = self._transfer_variable.local_party() if local_party is None else local_party
        self._all_parties = self._transfer_variable.all_parties() if all_parties is None else all_parties
        self._party_idx = self._all_parties.index(self._local_party)
        self._other_parties = self._all_parties[:self._party_idx] + self._all_parties[(self._party_idx + 1):]

    @property
    def party(self):
        return self._local_party

    @property
    def parties(self):
        return self._all_parties

    @property
    def other_parties(self):
        return self._other_parties

    @property
    def party_idx(self):
        return self._party_idx

    def remote_q_field(self, q_field, party):
        return self._q_field_variable.remote_parties(q_field, party, suffix=("q_field",))

    def get_q_field(self, party):
        return self._q_field_variable.get_parties(party, suffix=("q_field",))

    def get_rescontruct_shares(self, tensor_name):
        return self._rescontruct_variable.get_parties(self._other_parties, suffix=(tensor_name,))

    def broadcast_rescontruct_share(self, share, tensor_name):
        return self._rescontruct_variable.remote_parties(share, self._other_parties, suffix=(tensor_name,))

    def remote_share(self, share, tensor_name, party):
        return self._share_variable.remote_parties(share, party, suffix=(tensor_name,))

    def get_share(self, tensor_name, party):
        return self._share_variable.get_parties(party, suffix=(tensor_name,))

    def remote_encrypted_tensor(self, encrypted, tag):
        return self._mul_triplets_encrypted_variable.remote_parties(encrypted, parties=self._other_parties, suffix=tag)

    def remote_encrypted_cross_tensor(self, encrypted, parties, tag):
        return self._mul_triplets_cross_variable.remote_parties(encrypted, parties=parties, suffix=tag)

    def get_encrypted_tensors(self, tag):
        return (self._other_parties,
                self._mul_triplets_encrypted_variable.get_parties(parties=self._other_parties, suffix=tag))

    def get_encrypted_cross_tensors(self, tag):
        return self._mul_triplets_cross_variable.get_parties(parties=self._other_parties, suffix=tag)

    def clean(self):
        self._rescontruct_variable.clean()
        self._share_variable.clean()
        self._rescontruct_variable.clean()
        self._mul_triplets_encrypted_variable.clean()
        self._mul_triplets_cross_variable.clean()
        self._q_field_variable.clean()

    def set_flowid(self, flowid):
        self._transfer_variable.set_flowid(flowid)

```

### fate_arch/federation/transfer_variable.py

```PYTHON

```

### fate_arch/federation/transfer_variable.py

```PYTHON

```

### fate_arch/federation/transfer_variable.py

```PYTHON

```

