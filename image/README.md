# 代码运行说明

## 1 环境准备

### 1.1 硬件及操作系统条件

本队在平台提交的结果均在Windows 11系统上运行得到，具体配置如下：

- 处理器（CPU）：12th Gen Intel(R) Core(TM) i5-1240P   1.70 GHz
- 内存（RAM）：16.0 GB
- 操作系统（OS）：Windows 11 家庭版，23H2，64位操作系统，基于x64的处理器

（注：本解决方案无需使用GPU）

在复现阶段，我们使用Linux云服务器进行运行、测试、打包，具体配置如下：

- 处理器（CPU）：Intel Ice Lake 2.8GHz
- 内存（RAM）：32.0 GB
- 操作系统（OS）：Ubuntu 22.04 server 64bit

**主办方如果使用Linux运行代码请尽量保证满足以下要求**：

- 内存（RAM）>=32.0 GB

不然可能会出现在运行代码时因为内存不足而被终止（Killed）。

### 1.2 环境依赖

本队在平台提交的结果均在Windows 11系统上运行得到，该计算机使用的Python环境为

```
python 3.8.18
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.3
lightgbm==4.1.0
```

在复现阶段，我们使用Linux下的docker镜像，其Python环境为

```
python 3.7.17
numpy==1.21.6
pandas==1.3.5
matplotlib==3.5.3
lightgbm==4.1.0
```

## 2 代码运行步骤

### 2.1 启动docker镜像

我们提供了Ubuntu22.04下的docker镜像。请按以下步骤启动镜像：

启动docker服务

```shell
sudo service docker start
```

请保证当前在项目文件夹下（当前文件夹包含`data`和`image`两个文件夹），加载docker镜像文件

```shell
sudo docker load -i ./image/bdci2023-jd.tar
```

运行镜像

```shell
sudo docker run -v .:/root -it bdci2023-jd
```

### 2.2 运行代码

因为在运行镜像时进行了路径映射，所以在镜像中首先需要进入项目所在位置

```shell
cd /root
```

运行`run.sh`即可开始预处理、训练、预测（**注意不要进到image文件夹下运行！**）

```shell
sh ./image/run.sh
```

## 3 额外说明

### 3.1 `run.sh`中的运行步骤

进入代码文件夹

```shell
cd ./data/code/
```

预处理所有数据并整合成训练集及测试集

```shell
python preprocess.py
```

使用预处理后的文件进行训练

```shell
python train.py
```

使用模型进行预测并整合得到结果

```shell
python test.py
```

### 3.2 其他

**运行结果与提交结果不同属正常现象**。因为本队在提交阶段均使用Windows进行编程，而在复现阶段需使用Linux，系统间的某些底层计算处理方式有所不同，同时提交和复现阶段使用的环境也有所不同（Python及依赖库版本有所不同），所以结果出现略微偏差实属正常。需要注意的是，我们在训练阶段均设置了随机数种子（seed），并通过测试发现在同样环境下每次运行结果均相同，因此不是算法本身的随机性导致的偏差。