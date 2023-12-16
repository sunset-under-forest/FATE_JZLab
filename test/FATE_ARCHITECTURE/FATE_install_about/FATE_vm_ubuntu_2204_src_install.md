2023年7月23日

## 原先的测试环境是在 wsl 上的，发现存在分配的内存占用过多、cpu 经常跑满的问题，所以决定在虚拟机上重新部署环境

虚拟机环境：Ubuntu 22.04.02

### 虚拟机配置过程

```bash
# 对应镜像下载地址 https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/jammy/ubuntu-22.04.2-desktop-amd64.iso

# 网络配置NAT，分配CPU 4核，内存 8G，硬盘 60G，默认bios启动

# 磁盘分配：/boot 500M，/ 30G，swap 8G，/home REST

# 按常例换源，安装基本工具后再安装docker
# Docker version 20.10.21, build 20.10.21-0ubuntu1~22.04.3

# 安装docker后，按照上面的步骤创建容器并进行测试
# fate_test suite -i examples/dsl/v2/homo_nn


```

<hr>

2023 年 7 月 25 日

## 源码分析

### 从源码部署 FATE 单机版开始入手，以下是我使用的配置命令

教程地址：https://github.com/FederatedAI/FATE/blob/master/deploy/standalone-deploy/doc/standalone_fate_source_code_deployment_guide.zh.md

#### 1. 检测本地 8080、9360、9380 端口是否被占用

```bash
netstat -apln|grep 8080 
netstat -apln|grep 9360 
netstat -apln|grep 9380
```

| 组件      | 端口      | 说明                       |
| --------- | --------- | -------------------------- |
| fate_flow | 9360;9380 | 联合学习任务流水线管理模块 |
| fateboard | 8080      | 联合学习过程可视化模块     |

#### 2. 获取源代码

教程里面获取源代码说要用 git 来 clone 仓库，但是不管是git还是网页
下载的包中相关的子模块都是空的，例如 fateflow 和 fateboard 等，所以这些子仓库模块要自行下载补充。

```bash
cd /home/lab/federated_learning/fate/from_src_build/FATE # 进入代码的存放目录
export FATE_PROJECT_BASE=$PWD
export version=`grep "FATE=" ${FATE_PROJECT_BASE}/fate.env | awk -F "=" '{print $2}'`
```

#### 3. 安装并配置Python环境

```bash
python3 -m venv fate_venv # {虚拟环境名称}
export FATE_VENV_BASE=$FATE_PROJECT_BASE/fate_venv  # {放置虚拟环境的根目录}/{虚拟环境名称}
source ${FATE_VENV_BASE}/bin/activate   # 激活虚拟环境

sudo bash bin/install_os_dependencies.sh    # 安装依赖
sudo apt-get install python3.8-dev  # 防止安装python依赖库时报错
pip install -r python/requirements.txt   # 安装python依赖库
```

2023年7月26日
## 承接上面的源码部署，继续分析


#### 4. 配置FATE 
编辑bin/init_env.sh环境变量文件,将FATE_PROJECT_BASE和FATE_VENV_BASE的值替换为实际值
```bash
cd ${FATE_PROJECT_BASE}
sed -i.bak "s#PYTHONPATH=.*#PYTHONPATH=$PWD/python:$PWD/fateflow/python#g" bin/init_env.sh
sed -i.bak "s#venv=.*#venv=${FATE_VENV_BASE}#g" bin/init_env.sh
```

#### 5. 启动fate flow server
```bash 
cd ${FATE_PROJECT_BASE}
source bin/init_env.sh

cd fateflow
bash bin/service.sh status
bash bin/service.sh start
```

#### 6. 安装fate client
```bash
cd ${FATE_PROJECT_BASE}
source bin/init_env.sh

cd python/fate_client/
python setup.py install
```

初始化`fate flow client`
```bash
cd ../../
flow init -c conf/service_conf.yaml
```

##### 在这个过程中，我分析了fate_client的安装过程，也就是${FATE_PROJECT_BASE}/python/fate_client/setup.py文件，正好没有了解过python的打包和安装，所以就顺便学习了一下setuptools的用法

在这个安装过程中，setup.py文件中的setup函数会调用setuptools.setup函数，根据传入的参数来进行安装，这里列出了一些相关参数的说明：

```python   
# packages 指定要打包安装的包
# package_data 指定要打包安装的包中包含的数据文件（除了.py文件之外的文件），文件中的是默认包含包内所有文件
# install_requires 指定依赖的包，若依赖的包不在本地，则会从指定的源下载
# entry_points 指定安装后的可执行文件，例如这里的flow和pipeline
```

经过测试我还额外发现了entry_points指定安装的可执行文件在windows系统中会编译为.exe二进制文件，而在linux系统中，表面上是可执行文件，但是实际内容是python脚本。

#### 7. 编译fateboard
```bash
mkdir ${FATE_PROJECT_BASE}/env
cd ${FATE_PROJECT_BASE}/env

wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/resources/jdk-8u192.tar.gz
tar xzf jdk-8u192.tar.gz

wget https://dlcdn.apache.org/maven/maven-3/3.8.8/binaries/apache-maven-3.8.8-bin.tar.gz
tar xzf apache-maven-3.8.8-bin.tar.gz

export JAVA_HOME=${FATE_PROJECT_BASE}/env/jdk-8u192

cd ${FATE_PROJECT_BASE}
sed -i.bak "s#JAVA_HOME=.*#JAVA_HOME=${JAVA_HOME}#g" bin/init_env.sh


cd ${FATE_PROJECT_BASE}/fateboard
${FATE_PROJECT_BASE}/env/apache-maven-3.8.8/bin/mvn -DskipTests clean package # 最好给maven换个源，不然会很慢，还有可能出错

# 由于官方给的教程又和实际不符，所以我加了一些改动


cd ${FATE_PROJECT_BASE}/fateboard
mkdir conf
cp src/main/resources/application.properties conf/application.properties
cp bin/service.sh service.sh
ln -s  target/fateboard-1.11.1.jar fateboard.jar
bash service.sh status
bash service.sh start
```

由于fateboard只是一个用java做的可视化的工具，跟核心逻辑关系不太大，所以目前在这里我就没有分析它的源码了。

#### 至此源代码部署完成，可以进行测试了

```bash
# Toy测试
flow test toy -gid 10000 -hid 10000

# 单元测试
cd ${FATE_PROJECT_BASE}
bash ./python/federatedml/test/run_test.sh

```
```
OK
there are 0 failed test
```
源代码部署成功，测试通过


#### 分析教程可以知道，与 docker 部署不同的是，从源代码部署要自行设置环境变量和安装工具，分析安装脚本可知：

```
FATE_PROJECT_BASE 是项目文件的根路径
FATE_VENV_BASE 是执行python命令的虚拟环境根路径

bin/init_env.sh 是fate框架用来初始化环境的

bin/install_os_dependencies.sh 是fate框架用来为安装库提供依赖的

python/requirements.txt 中记录了fate框架需要的依赖



```


#### service.sh:

##### start 函数的逻辑如下：
```
1. 首先调用 getpid 函数获取当前进程的 PID。
2. 如果当前进程的 PID 为空，则创建日志目录并启动 fate_flow_server.py 进程。
3. 如果传入的第一个参数为 "front"，则在前台启动 fate_flow_server.py 进程，否则在后台启动。
4. 等待进程启动，最多等待 10 秒钟。
5. 如果进程启动成功，则输出进程的 PID。
6. 如果进程启动失败，则输出错误信息。

在启动 fate_flow_server.py 进程之前，会设置环境变量 FATE_PROJECT_BASE 为 ${PROJECT_BASE}，并在进程启动后再将其删除。同时，进程的标准输出和标准错误输出会被重定向到 ${log_dir}/console.log 和 ${log_dir}/error.log 文件中。
```

`可知fate flow server的入口点就是${FATE_FLOW_BASE}/python/fate_flow/fate_flow_server.py`

`参与方提交数据和任务的命令行工具flow和pipeline的入口点分别是$FATE_PROJECT_BASE/python/fate_client/flow_client/flow.py和$FATE_PROJECT_BASE/python/fate_client/pipeline/pipeline_cli.py`
