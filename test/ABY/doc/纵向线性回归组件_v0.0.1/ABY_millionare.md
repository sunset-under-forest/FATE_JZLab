# ABY框架百万富翁测试组件嵌入FATE框架记录

进入FATE框架算法包下

```bash
export FEDERATEDML_LIBRARY_PATH=/home/lab/federated_learning/fate/from_src_build/FATE/python/federatedml # {federatedml包路径}
cd $FEDERATEDML_LIBRARY_PATH
mkdir -p ABY && cd ABY
mkdir -p CPP && cd CPP
mkdir -p extern && cd extern
```

将ABY项目源码复制到此处，目录名为ABY，注意子项目库也不能漏

## 测试ABY框架的基本使用

```bash
export ABY_LIBRARY_PATH=/home/lab/federated_learning/fate/from_src_build/FATE/python/federatedml/ABY/CPP/extern/ABY # {ABY项目根目录}
cd $ABY_LIBRARY_PATH
```

aby项目根目录结构

```
$ABY_LIBRARY_PATH
├── bin
├── cmake
├── CMakeLists.txt
├── Doxyfile
├── extern
├── LICENSE
├── README.md
├── runtest_scr.sh
└── src		#	源码目录包
```

创建build目录，并在此构建项目

```bash
mkdir -p build && cd build
cmake .. -DABY_BUILD_EXE=On
make
```

会有很多警告，只要不报错就没问题

进行基本功能测试

```bash
cd bin
./millionaire_prob_test -r 0 & ./millionaire_prob_test -r 1
```

此过程没有问题即可进行开发

## C++层开发

```bash
export ABY_CPP_SRC_PATH=$FEDERATEDML_LIBRARY_PATH/ABY/CPP
cd $ABY_CPP_SRC_PATH
vim CMakeLists.txt
```

CMakeLists.txt内容

```cmake
cmake_minimum_required(VERSION 3.12)
project(FATE_ABY LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # fpic
set(ABY_PATH "{ABY的根路径}")
find_package(ABY QUIET)
if(ABY_FOUND)
        message(STATUS "Found ABY")
elseif (NOT ABY_FOUND AND NOT TARGET ABY::aby)
        message("ABY was not found: add ABY subdirectory")
        add_subdirectory(extern/ABY)
endif()

add_subdirectory(src/millionaire_prob_test)
```

```bash
mkdir -p src/millionaire_prob_test && cd src/millionaire_prob_test
vim CMakeLists.txt
```

CMakeLists.txt内容

```cmake
add_executable(FATE_ABY_millionaire_prob_test FATE_ABY_millionaire_prob_test.cpp
        FATE_ABY_millionaire_prob_test.h)
target_link_libraries(FATE_ABY_millionaire_prob_test ABY::aby ENCRYPTO_utils::encrypto_utils)
add_library(FATE_ABY_millionaire_prob_test_lib SHARED FATE_ABY_millionaire_prob_test.cpp
        FATE_ABY_millionaire_prob_test.h)
target_link_libraries(FATE_ABY_millionaire_prob_test_lib ABY::aby ENCRYPTO_utils::encrypto_utils)
```

```bash
vim FATE_ABY_millionaire_prob_test.cpp
vim FATE_ABY_millionaire_prob_test.h
```

FATE_ABY_millionaire_prob_test.cpp内容

```cpp
#include "FATE_ABY_millionaire_prob_test.h"

share *BuildMillionaireProbCircuit(share *s_alice, share *s_bob,
                                   BooleanCircuit *bc) {

    share *out;

    /** Calling the greater than equal function in the Boolean circuit class.*/
    out = bc->PutGTGate(s_alice, s_bob);

    return out;
}


int32_t my_test_millionaire(u_int32_t money, e_role role, const std::string &address, uint16_t port, seclvl seclvl,
                            uint32_t bitlen, uint32_t nthreads, e_mt_gen_alg mt_alg, e_sharing sharing) {


    ABYParty *party = new ABYParty(role, address, port, seclvl, bitlen, nthreads,
                                   mt_alg);
    /*
     * role: the role of the party, SERVER or CLIENT，如果是SERVER，则会在本地端口上建立一个socket，监听来自CLIENT的连接，如果是CLIENT，则会连接到SERVER的socket上
     * */

    std::vector<Sharing *> &sharings = party->GetSharings();

    Circuit *circ = sharings[sharing]->GetCircuitBuildRoutine();

    share *s_alice_money, *s_bob_money, *s_out;

    uint32_t output;

    if (role == SERVER) {    // SERVER == 0, CLIENT == 1
        s_alice_money = circ->PutDummyINGate(bitlen);
        s_bob_money = circ->PutINGate(money, bitlen, SERVER);   // SEVER is the role of Bob
    } else { //role == CLIENT
        s_alice_money = circ->PutINGate(money, bitlen, CLIENT);   // CLIENT is the role of Alice
        s_bob_money = circ->PutDummyINGate(bitlen);
    }

    s_out = BuildMillionaireProbCircuit(s_alice_money, s_bob_money,
                                        (BooleanCircuit *) circ);

    s_out = circ->PutOUTGate(s_out, ALL);

    party->ExecCircuit();

    output = s_out->get_clear_value<uint32_t>();

    std::cout << "Testing Millionaire's Problem in " << get_sharing_name(sharing)
              << " sharing: " << std::endl;
    if (role == SERVER) {
        std::cout << "\nBob's Money:\t" << money;
        std::cout << "\nCircuit Result:\t" << (output ? ALICE : BOB);
        std::cout << "\nOutput\t" << output;
    } else {
        std::cout << "\nAlice's Money:\t" << money;
        std::cout << "\nCircuit Result:\t" << (output ? ALICE : BOB);
        std::cout << "\nOutput\t" << output;

    }

    delete party;
    if (output == 1) {
        return 1;
    } else if (output == 0) {
        return 0;
    } else {
        return -1;
    }
}


extern "C"
int bob(uint32_t money , const char *address, uint16_t port){
    return my_test_millionaire(money, SERVER, address, port, get_sec_lvl(128), 32, 1, MT_OT, S_YAO);
}

extern "C"
int alice(uint32_t money , const char *address, uint16_t port){
    return my_test_millionaire(money, CLIENT, address, port, get_sec_lvl(128), 32, 1, MT_OT, S_YAO);
}

extern "C"
const char* string_test(const char *str){
    std::cout << str << std::endl;
    std::cout << "used!"<< std::endl;
    return "hello world";
}


extern "C" MY_LIB_API
int test() {
    uint32_t bitlen = 32, secparam = 128, nthreads = 1;
    uint16_t port = 7766;
    const char * address = "192.168.210.135";	// 改成自己的ip


    uint32_t bob_money, alice_money;
    srand(time(NULL));
    bob_money = rand() % 100;
    alice_money = rand() % 100;
    while (bob_money <= alice_money) {
        bob_money = rand() % 100;
        alice_money = rand() % 100;
    }

    std::cout << "True Result: " << (bob_money > alice_money ? "BOB" : "ALICE") << std::endl;
    pid_t pid = fork();
    if (pid < 0) {
        std::cout << "fork error" << std::endl;
        exit(1);
    }
    if (pid == 0) {
        // 子进程
        bob(bob_money, "0.0.0.0", port);
    } else {
        // 父进程
        alice(alice_money, address, port);
    }
    return 0;
}

int main(){
    test();
}
```

FATE_ABY_millionaire_prob_test.h内容

```cpp
//
// Created by 0ne_bey0nd on 2023/10/6.
//

#ifndef ABY_FATE_ABY_MILLIONAIRE_PROB_TEST_H
#define ABY_FATE_ABY_MILLIONAIRE_PROB_TEST_H
#include <ENCRYPTO_utils/crypto/crypto.h>
#include <ENCRYPTO_utils/parse_options.h>
#include <iostream>
#include "../../extern/ABY/src/abycore/circuit/booleancircuits.h"
#include "../../extern/ABY/src/abycore/circuit/arithmeticcircuits.h"
#include "../../extern/ABY/src/abycore/circuit/circuit.h"
#include "../../extern/ABY/src/abycore/aby/abyparty.h"
#include "../../extern/ABY/src/abycore/sharing/sharing.h"

#if defined(_MSC_VER)
#define MY_LIB_API __declspec(dllexport) // Microsoft
#elif defined(__GNUC__)
#define MY_LIB_API __attribute__((visibility("default"))) // GCC    # default代表外部可见，hidden代表外部不可见，就是public和private的意思
#else
#define MY_LIB_API // Most compilers export all the symbols by default. We hope for the best here.
#pragma warning Unknown dynamic link import/export semantics.
#endif
#endif //ABY_FATE_ABY_MILLIONAIRE_PROB_TEST_H

#define ALICE   "ALICE"
#define BOB     "BOB"

```

编译测试

```bash
cd $ABY_CPP_SRC_PATH
mkdir  -p build && cd build
cmake ..
make
cd src/millionaire_prob_test
./FATE_ABY_millionaire_prob_test
```

没有出现问题，开始python测试动态链接库功能

## PYTHON层测试

在刚刚的目录下

```bash
vim test.py
vim test_bob.py
vim test_alice.py
```

test.py

```python
from ctypes import *
import ctypes
dll = CDLL("./libFATE_ABY_millionaire_prob_test_lib.so")
dll.test()

# string_test , ctypes 中只能传最基础的char* ， 传std::string的话很难
# address = b"127.0.0.1"
# dll.string_test.argtype = ctypes.c_char_p
# dll.string_test.restype = ctypes.c_char_p
# print(dll.string_test(address))

```

test_bob.py

```python
from ctypes import *
dll = CDLL("./libFATE_ABY_millionaire_prob_test_lib.so")
address = b"127.0.0.1"
result = dll.bob(40,address,7766)
print(f"Result:\t{result}")
```

test_alice.py

```python
from ctypes import *
dll = CDLL("./libFATE_ABY_millionaire_prob_test_lib.so")
address = b"127.0.0.1"
result = dll.alice(56,address,7766)
print(f"Result:\t{result}")
```

执行测试命令

```bash
python test_bob.py & python test_alice.py
```

没有出现问题，开始python开发FATE组件

## PYTHON层开发

声明变量

```bash
export ABY_COMPONENT_PATH=$FEDERATEDML_LIBRARY_PATH/ABY
```

进入目录并创建lib目录，存放刚才的编译好的动态链接库

```bash
cd $ABY_COMPONENT_PATH
mkdir -p lib
cp $ABY_CPP_SRC_PATH/build/src/millionaire_prob_test/libFATE_ABY_millionaire_prob_test_lib.so lib/
```

创建python包

```bash
mkdir -p millionaire_prob_test
touch millionaire_prob_test/__init__.py
```

开始编写测试组件

```bash
cd $FEDERATEDML_LIBRARY_PATH
# 组件类
vim ABY/millionaire_prob_test/aby_millionaire_prob_test.py

# 组件参数类
vim param/aby_millionaire_prob_test_param.py

# 组件反射绑定文件
vim components/aby_millionaire_prob_test.py

```

ABY/millionaire_prob_test/aby_millionaire_prob_test.py

```python
from federatedml.model_base import ModelBase
from federatedml.param.aby_millionaire_prob_test_param import ABYMillionaireProbTestParam
from federatedml.util import LOGGER
import ctypes
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
ABY_PKG_PATH = os.path.join(CURRENT_PATH, "..")
DLL_PATH = os.path.join(ABY_PKG_PATH, "lib")


class ABYMillionaireProbTest(ModelBase):
    def __init__(self):
        super().__init__()
        self.dll = None
        self.load_dll()

    def load_dll(self):
        self.dll = ctypes.CDLL(os.path.join(DLL_PATH, "libFATE_ABY_millionaire_prob_test_lib.so"))


class ABYMillionaireProbTestGuest(ABYMillionaireProbTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYMillionaireProbTestGuest'
        self.model_param = ABYMillionaireProbTestParam()

    def fit(self, train_data, validate_data=None):
        """
        测试
        """
        LOGGER.info("Start ABY Millionaire Prob Test Guest")
        money = self.model_param.money
        aby_role = self.model_param.aby_role
        LOGGER.info("aby_role: {}".format(aby_role))
        if aby_role!= "server":
            raise ValueError("aby_role should be server as bob")
        LOGGER.info("So this is BOB and BOB's money: {}".format(money))
        address = self.model_param.address
        port = self.model_param.port
        LOGGER.info("address: {}".format(address))
        LOGGER.info("port: {}".format(port))
        LOGGER.debug("dll: {}".format(self.dll))
        result = self.dll.bob(money, address.encode(), port)

        # 这里的百万富翁例子是判断ALICE的钱是否大于BOB的钱，也就是说如果result为1，说明ALICE的钱比BOB多，为0则相反
        LOGGER.info("result: {}".format(result))
        if result == 1:
            LOGGER.info("ALICE is richer than BOB")
        elif result == 0:
            LOGGER.info("BOB is richer than ALICE")

        return result


class ABYMillionaireProbTestHost(ABYMillionaireProbTest):
    def __init__(self):
        super().__init__()
        self.model_name = 'ABYMillionaireProbTestHost'
        self.model_param = ABYMillionaireProbTestParam()	

    def fit(self, train_data, validate_data=None):
        """
        测试
        """
        LOGGER.info("Start ABY Millionaire Prob Test Host")
        money = self.model_param.money
        aby_role = self.model_param.aby_role
        LOGGER.info("aby_role: {}".format(aby_role))
        if aby_role!= "client":
            raise ValueError("aby_role should be client as alice")
        LOGGER.info("So this is ALICE and ALICE's money: {}".format(money))
        address = self.model_param.address
        port = self.model_param.port
        LOGGER.info("address: {}".format(address))
        LOGGER.info("port: {}".format(port))
        LOGGER.debug("dll: {}".format(self.dll))
        result = self.dll.alice(money, address.encode(), port)

        # 这里的百万富翁例子是判断ALICE的钱是否大于BOB的钱，也就是说如果result为1，说明ALICE的钱比BOB多，为0则相反
        LOGGER.info("result: {}".format(result))
        if result == 1:
            LOGGER.info("ALICE is richer than BOB")
        elif result == 0:
            LOGGER.info("BOB is richer than ALICE")
        return result


```

param/aby_millionaire_prob_test_param.py

```python
import random

from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER

class ABYMillionaireProbTestParam(BaseParam):
    """
    ABY Millionaire Prob Test

    Parameters
    ----------
    aby_role : str, default: "server"
        Specify the role of this party.
    money : int, default: a random integer between 1 and 100
        Specify the money of this party.
    address : str, default: "0.0.0.0"
        Specify the address of this party.
    port : int, default: 7766
        Specify the port of this party.
    """

    def __init__(self, aby_role="server" , money=random.randint(1,100), address="0.0.0.0", port=7766):
        super(ABYMillionaireProbTestParam, self).__init__()       # super() 函数是用于调用父类(超类)的一个方法。
        self.aby_role = aby_role.lower().strip()
        self.money = money
        self.address = address
        self.port = port

    def check(self):
        model_param_descr = "ABY Millionaire Prob Test param's "
        if self.aby_role is not None:
            if not isinstance(self.aby_role, str):
                raise ValueError(f"{model_param_descr} role should be str type")
            if self.aby_role not in ["server", "client"]:
                raise ValueError(f"{model_param_descr} role should be 'server' or 'client'")

        if self.money is not None:
            BaseParam.check_nonnegative_number(self.money, f"{model_param_descr} money ")
            if not isinstance(self.money, int):
                raise ValueError(f"{model_param_descr} money should be int type")
            if self.money > 0xffffffff:
                raise ValueError(f"{model_param_descr} money should be less than 0xffffffff")

        if self.address is not None:
            if not isinstance(self.address, str):
                raise ValueError(f"{model_param_descr} address should be str type")
            # 检查是否符合ip地址格式
            import re
            if any(map(lambda n: int(n) > 255, re.match(r'^(\d+)\.(\d+)\.(\d+)\.(\d+)$', self.address).groups())):
                raise ValueError(f"{model_param_descr} address should be ip address format" )


        if self.port is not None:
            if not isinstance(self.port, int):
                raise ValueError(f"{model_param_descr} port should be int type")
            BaseParam.check_positive_integer(self.port, f"{model_param_descr} port ")
            if self.port > 65535:
                raise ValueError(f"{model_param_descr} port should be less than 65535")


```

components/aby_millionaire_prob_test.py

```python
from .components import ComponentMeta

aby_millionaire_prob_test_cpn_meta = ComponentMeta("ABYMillionaireProbTest")

@aby_millionaire_prob_test_cpn_meta.bind_param
def aby_millionaire_prob_test_param():
    from federatedml.param.aby_millionaire_prob_test_param import ABYMillionaireProbTestParam

    return ABYMillionaireProbTestParam


@aby_millionaire_prob_test_cpn_meta.bind_runner.on_guest
def aby_millionaire_prob_test_runner_guest():
    from federatedml.ABY.millionaire_prob_test.aby_millionaire_prob_test import ABYMillionaireProbTestGuest

    return ABYMillionaireProbTestGuest


@aby_millionaire_prob_test_cpn_meta.bind_runner.on_host
def aby_millionaire_prob_test_runner_host():
    from federatedml.ABY.millionaire_prob_test.aby_millionaire_prob_test import ABYMillionaireProbTestHost

    return ABYMillionaireProbTestHost
```

## 测试

首先重启fateflow

重启完成后输入命令查找组件是否已被注册

```bash
flow provider list|grep "abymillionaireprobtest" -n

# (fate_venv) lab@lab-virtual-machine:~/federated_learning/fate/from_src_build/FATE/fateflow$ flow provider list|grep "abymillionaireprobtest" -n
# 55:                    "abymillionaireprobtest",

```

已经注册组件成功后，准备测试相关文件

```bash
cd $FATE_PROJECT_BASE
mkdir -p aby_fate_test && cd aby_fate_test
export ABY_FATE_TEST_PATH=`pwd`
mkdir -p millionaire_prob_test
```
准备数据
```bash
mkdir -p data
# 从FATE提供的数据中随便选择一个复制过来，因为在我们的测试组件里面没有用到数据
cp $FATE_PROJECT_BASE/examples/data/unittest_data.csv  data/millionaire_prob_test.csv
cp data/millionaire_prob_test.csv data/millionaire_prob_test_guest.csv
cp data/millionaire_prob_test.csv data/millionaire_prob_test_host.csv
```

上传数据

```bash
vim millionaire_prob_test/upload_data_guest.py
vim millionaire_prob_test/upload_data_host.py
```

upload_data_guest.py（注意更换目录）

```python
import os
import argparse
from pipeline.backend.pipeline import PipeLine

DATA_BASE = "/home/lab/federated_learning/fate/from_src_build/FATE/aby_fate_test/data"  # 数据的存放的目录


def main(data_base=DATA_BASE):
    # parties config
    guest = 10000

    # partition for data storage
    partition = 4

    data = {"name": "millionaire_prob_test_guest", "namespace": f"test"}

    pipeline_upload = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "millionaire_prob_test_guest.csv"),
                                    table_name=data["name"],             # table name
                                    namespace=data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.upload(drop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--base", "-b", type=str,
                        help="data base, path to directory that contains examples/data")

    args = parser.parse_args()
    if args.base is not None:
        main(args.base)
    else:
        main()

```

upload_data_host.py（注意更换目录）

```python
import os
import argparse
from pipeline.backend.pipeline import PipeLine

DATA_BASE = "/home/lab/federated_learning/fate/from_src_build/FATE/aby_fate_test/data"  # 数据的存放的目录


def main(data_base=DATA_BASE):
    # parties config
    host = 10000

    # partition for data storage
    partition = 4

    data = {"name": "millionaire_prob_test_host", "namespace": f"test"}

    pipeline_upload = PipeLine().set_initiator(role="host", party_id=host).set_roles(host=host)

    pipeline_upload.add_upload_data(file=os.path.join(data_base, "millionaire_prob_test_host.csv"),
                                    table_name=data["name"],             # table name
                                    namespace=data["namespace"],         # namespace
                                    head=1, partition=partition)               # data info

    pipeline_upload.upload(drop=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--base", "-b", type=str,
                        help="data base, path to directory that contains examples/data")

    args = parser.parse_args()
    if args.base is not None:
        main(args.base)
    else:
        main()

```

模拟guest上传

```bash
python millionaire_prob_test/upload_data_guest.py
```

模拟host上传

```bash
python millionaire_prob_test/upload_data_host.py
```

编写任务文件

```bash
vim millionaire_prob_test/millionaire_prob_test_conf.json
vim millionaire_prob_test/millionaire_prob_test_dsl.json
```

millionaire_prob_test/millionaire_prob_test_conf.json（如果不是单机测试的话注意地址和端口要修改）

```json
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
                    "aby_millionaire_prob_test_0": {
                        "aby_role": "client",
                        "money": 66
                    },
                    "reader_0": {
                        "table": {
                            "name": "millionaire_prob_test_host",
                            "namespace": "test"
                        }
                    }
                }
            },
            "guest": {
                "0": {
                    "aby_millionaire_prob_test_0": {
                        "aby_role": "server",
                        "money": 77
                    },
                    "reader_0": {
                        "table": {
                            "name": "millionaire_prob_test_guest",
                            "namespace": "test"
                        }
                    }
                }
            }
        },
        "common": {
            "aby_millionaire_prob_test_0": {
                "address": "0.0.0.0",
                "port": 7766
            }
        }
    }
}
```

millionaire_prob_test/millionaire_prob_test_dsl.json

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
        "aby_millionaire_prob_test_0": {
            "module": "ABYMillionaireProbTest",
            "input": {
                "data": {
                    "train_data": [
                        "reader_0.data"
                    ]
                }
            },
            "output": {
                "data": [
                    "data"
                ]
            }
        }
    }
}
```

根据FATE框架job dsl规则可知，这里一定要首先使用Reader读取数据，有了数据的传递FATE才能识别到并执行组件的fit方法，详情在FATE框架任务执行底层原理会有具体介绍。

发布任务

```bash
flow job submit -c millionaire_prob_test/millionaire_prob_test_conf.json -d millionaire_prob_test/millionaire_prob_test_dsl.json 
```

通过fateboard查看日志

host

```
432
[INFO] [2023-10-11 17:03:42,941] [202310111703370130760] [51953:140673322897408] - [aby_millionaire_prob_test.fit] [line:66]: Start ABY Millionaire Prob Test Host
433
[INFO] [2023-10-11 17:03:42,941] [202310111703370130760] [51953:140673322897408] - [aby_millionaire_prob_test.fit] [line:69]: aby_role: client
434
[INFO] [2023-10-11 17:03:42,941] [202310111703370130760] [51953:140673322897408] - [aby_millionaire_prob_test.fit] [line:72]: So this is ALICE and ALICE's money: 66
435
[INFO] [2023-10-11 17:03:42,941] [202310111703370130760] [51953:140673322897408] - [aby_millionaire_prob_test.fit] [line:75]: address: xxx.xxx.xxx.xxx
436
[INFO] [2023-10-11 17:03:42,941] [202310111703370130760] [51953:140673322897408] - [aby_millionaire_prob_test.fit] [line:76]: port: 7766
437
[INFO] [2023-10-11 17:03:43,104] [202310111703370130760] [51953:140673322897408] - [aby_millionaire_prob_test.fit] [line:81]: result: 0
438
[INFO] [2023-10-11 17:03:43,104] [202310111703370130760] [51953:140673322897408] - [aby_millionaire_prob_test.fit] [line:85]: BOB is richer than ALICE
```

guest

```
432
[INFO] [2023-10-11 17:03:42,835] [202310111703370130760] [51950:140212784574464] - [aby_millionaire_prob_test.fit] [line:32]: Start ABY Millionaire Prob Test Guest
433
[INFO] [2023-10-11 17:03:42,835] [202310111703370130760] [51950:140212784574464] - [aby_millionaire_prob_test.fit] [line:35]: aby_role: server
434
[INFO] [2023-10-11 17:03:42,835] [202310111703370130760] [51950:140212784574464] - [aby_millionaire_prob_test.fit] [line:38]: So this is BOB and BOB's money: 77
435
[INFO] [2023-10-11 17:03:42,835] [202310111703370130760] [51950:140212784574464] - [aby_millionaire_prob_test.fit] [line:41]: address: xxx.xxx.xxx.xxx
436
[INFO] [2023-10-11 17:03:42,835] [202310111703370130760] [51950:140212784574464] - [aby_millionaire_prob_test.fit] [line:42]: port: 7766
437
[INFO] [2023-10-11 17:03:43,104] [202310111703370130760] [51950:140212784574464] - [aby_millionaire_prob_test.fit] [line:47]: result: 0
438
[INFO] [2023-10-11 17:03:43,104] [202310111703370130760] [51950:140212784574464] - [aby_millionaire_prob_test.fit] [line:51]: BOB is richer than ALICE
```

> ip那里的xxx.xxx.xxx.xxx是fate框架的log显示里有自动屏蔽ip地址的功能

可以看到已经测试成功，双方在不知道对方数字的情况下比较出了大小。
