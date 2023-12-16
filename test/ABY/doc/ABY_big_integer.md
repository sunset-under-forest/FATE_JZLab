# ABY大整数接口方案

参考python大整数实现

> https://www.cnblogs.com/traditional/p/13437243.html

博客里写到：

PYTHON 的 INT 类型结构体可表示如下

```C
//如果把这个PyLongObject更细致的展开一下就是
typedef struct {
    Py_ssize_t ob_refcnt; //引用计数
    struct _typeobject *ob_type; //类型
    Py_ssize_t ob_size; //维护的元素个数
    digit ob_digit[1]; //digit类型的数组,长度为1
} PyLongObject;
```

> 为什么只用32位无符号整型的30位（64位系统中），通过查阅好像是python源码规定这个使用位数是5的倍数，在32位系统中的ob_digit是16位整型，实际只用了15位。

```C
#if PyLong_SHIFT % 5 != 0
#error "longobject.c requires that PyLong_SHIFT be divisible by 5"
#endif
```

因为是做一个计算接口，所以我们只挑选需要的，这里就是`ob_size`和`ob_digit`

big_integer.py
```python



```

<br>

## 测试

将ABY构建的动态链接库赋值到lib文件夹下

