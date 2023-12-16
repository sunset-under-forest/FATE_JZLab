2023年8月8日
## spdz包分析——自下而上

### 从spdz包下的test目录中的guest测试开始，分析相关调用链

#### 1. spdz包下的test目录中的guest测试

**guest.py**
这个文件对SPDZ的调用是这一部分
``````python
with SPDZ() as spdz:
    x = FixedPointTensor.from_source("x", data)
    y = FixedPointTensor.from_source("y", partys[1])

    z = (x + y).get()
    t = (x - y).get()
    print(z)
    print(t)
``````

with语句会调用SPDZ类的__enter__方法，在with语句块结束时会调用__exit__方法，这两个方法都是在SPDZ类中定义的。

spdz.py
``````python
class SPDZ(object):
    __instance = None

    @classmethod
    def set_instance(cls, instance):
        prev = cls.__instance
        cls.__instance = instance
        return prev

    def __init__(self, name="ss", q_field=None, local_party=None, all_parties=None, use_mix_rand=False, n_length=1024):
        self.name_service = naming.NamingService(name)
        self._prev_name_service = None
        self._pre_instance = None



    def __enter__(self):
        self._prev_name_service = NamingService.set_instance(self.name_service)
        self._pre_instance = self.set_instance(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        NamingService.set_instance(self._pre_instance)
``````

再跳转到NamingService类中

naming.py

``````python
class NamingService(object):
    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            raise EnvironmentError("naming service not set")
        return cls.__instance

    @classmethod
    def set_instance(cls, instance):
        prev = cls.__instance
        cls.__instance = instance
        return prev

    def __init__(self, init_name="ss"):
        self._name = hashlib.md5(init_name.encode("utf-8")).hexdigest()

    def next(self):
        self._name = hashlib.md5(self._name.encode("utf-8")).hexdigest()
        return self._name
``````

可知在`with SPDZ() as spdz:`语句中，首先会创造一个SPDZ类的实例，调用__init__方法，实例化一个NamingService对象，调用NamingService类的__init__方法。然后在调用SPDZ类的__enter__方法时，会首先调用NamingService类的set_instance方法，将当前的NamingService对象赋值给NamingService类的__instance静态成员变量，然后调用SPDZ类的set_instance方法，将当前的SPDZ对象赋值给SPDZ类的__instance静态成员变量，然后返回当前的SPDZ对象。在with语句块结束时，会调用SPDZ类的__exit__方法，再一次调用本身的set_instance方法，将之前保存的SPDZ对象赋值给SPDZ类的__instance静态成员变量，实现在with语句内部使用SPDZ类跟SPDZ当前对象的绑定，set_instance方法类似MFC的画刷的SelectObject方法，传入新的，返回旧的。

