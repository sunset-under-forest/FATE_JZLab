## _部署 FATE 工业级联邦学习框架环境_

WSL 中 docker 部署
docker version: 20.10.21

### 执行过的命令

```bash
export version=1.11.2 # 指定版本

# sudo docker pull federatedai/standalone_fate:${version}  # 下载镜像
# 上面这条是官方给出的通过公共镜像服务拉取的镜像，但是拉取后发现跟文档前面要求的版本对不上，latest版本运行下来也有问题，所以采用通过镜像包的方法

# 下载镜像包
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/fate/${version}/release/standalone_fate_docker_image_${version}_release.tar.gz

# 加载镜像包
docker load -i standalone_fate_docker_image_${version}_release.tar.gz

# 查看镜像
docker images | grep federatedai/standalone_fate

# 启动容器
docker run -it --name standalone_fate -p 8080:8080 federatedai/standalone_fate:${version}
# 注意-it一定不能放在镜像名后面，不然会被当作启动参数，导致启动失败

# 直接可以进入容器
# [root@a097cd2b67b7 fate]#

# 测试
source bin/init_env.sh

# 功能测试
# Toy
flow test toy -gid 10000 -hid 10000

# success to calculate secure_sum, it is 1999.9999999999993

# 单元测试
fate_test unittest federatedml --yes
# there are 0 failed test


# fateboard面板 8080端口
# 默认用户名和密码都是admin


# TODO
# 在模拟实际应用场景的网络拓扑环境下通过虚拟机实现FATE的部署和测试应用


```

## 使用 docker 创建容器并通过 docker exec 进入正在运行容器的方法

```bash
docker exec -it standalone_fate bash
# 注意一定要-it ，只有i的话也能进入，但是没有tty，用起来有些问题，例如前面的命令行提示符不会显示
```

## 进入容器后的一些操作

```bash
# 更新yum源，https://mirrors.tuna.tsinghua.edu.cn/help/centos/
sed -e 's|^mirrorlist=|#mirrorlist=|g' \
         -e 's|^#baseurl=http://mirror.centos.org/centos|baseurl=https://mirrors.tuna.tsinghua.edu.cn/centos|g' \
         -i.bak \
         /etc/yum.repos.d/CentOS-*.repo

yum makecache
yum update

# 安装vim
yum install vim

# 将python3.8链接到/usr/bin/python3，也就是将python3.8添加到环境变量中
ln -snf /data/projects/fate/env/python/miniconda/bin/python3.8 /usr/bin/python3
# -n 不删除已存在的文件
```

2023 年 7 月 21 日

## vscode 连接 docker 容器进行开发调试的配置过程

### _首先前面配置了很多环境，但是发现现在的需求需要从头来过（vscode 调试需要让容器新增端口映射，需要在容器创建之前使用多个-p 参数指定端口，但是现在容器已经运行了，又不想直接删除容器重新配置）_

```bash
# 先关了之前的容器
docker stop standalone_fate # 或者 docker kill standalone_fate

# 通过docker commit命令将当前容器保存为镜像
docker commit standalone_fate my_fate

docker image ls
# 现在可以看到刚刚保存的镜像

# 通过docker run命令创建新的容器，指定多个端口映射
docker run -dit --name my_fate -p 5656:22 -p 8080:8080 my_fate
# -d参数是为了让容器在后台运行，不然会直接进入容器，而且通过ctrl+d退出容器后容器也会停止运行（也可以ctrl+p+q退出容器，这样不会停止）
# 创建的时候没有加-d的话也可以在exec进入容器的时候添加-d参数，但是不会进入容器，而是直接返回容器id，要进入容器还需要再次exec（直接-it进入）

```

本来想直接在容器里面开 ssh 服务连的，后来发现好像在容器里面开 ssh 服务并不安全，而且好像并没有很多人这样做，之后又发现 vscode 远程连接到服务器的时候可以直接附加到已经运行的容器上，所以就不用开 ssh 服务了，直接用 vscode 远程连接到容器上就行了