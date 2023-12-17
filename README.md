# 安联医盟——基于FATE+ABY的医疗数据高效联邦学习系统——v1.0.7

[![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Enabled-brightgreen)](https://www.example.com/federated-learning) [![Powered by JZLab](https://img.shields.io/badge/Powered%20by-JZLab-orange)](https://fate-oss.github.io/) [![Privacy-Focused](https://img.shields.io/badge/Privacy-Focused-blue)](https://www.example.com/privacy-policy) [![Version](https://img.shields.io/badge/Version-1.0.7-brightgreen)](https://www.example.com/releases) [![Documentation](https://img.shields.io/badge/Documentation-Yes-brightgreen)](https://www.example.com/docs) [![Contributors](https://img.shields.io/badge/Contributors-5-orange)](https://www.example.com/contributors)

安联医盟是一款基于FATE工业级联邦学习框架+ABY安全多方计算框架的医疗数据高效联邦学习系统，可以让医疗机构在保护数据安全和数据隐私的前提下进行数据协作。安联医盟项目在原FATE框架的基础上设计了
*基于数据增强框架的数据预处理方案*和*基于客户端贡献度的参数聚合算法组件*
，可以有效应对医疗场景下非独立同分布数据带来的挑战，提升联邦学习对用户数据的包容性并显著提升模型准确率等性能；本项目将ABY引入了FATE
，可以有效应对联邦学习过程通讯开销带来的挑战，大大提高了 FATE 框架进行联邦学习任务时的通信效率。系统具有高效、高性能、数据包容性强等优势，可广泛应用于疾病预测、医疗智能等场景。

<hr>

## 项目结构

```
test
├── ABY
│   ├── data
│   ├── README.md
│   ├── ...
├── data_enhancement
│   ├── data
│   ├── README.md
│   ├── ...
├── FedCon
│   ├── FedCon
│   ├── README.md
│   ├── ...
├── FATE_ARCHITECTURE
│   ├── README.md
│   ├── ...
├── ...
```

<hr>

## 模块文档

- [基于ABY计算框架的FATE安全多方计算底层嵌入](./test/ABY/README.md)
- [基于数据增强框架的数据预处理方案](./test/data_enhancement/README.md)
- [基于客户端贡献度参数聚合算法组件](./test/FedCon/README.md)
- [项目采用的医疗数据集](./test/data/README.md)

<hr>

## 总结

对于现阶段的联邦学习领域所面临的问题，本项目提出了以下解决方案：

| 现阶段联邦学习面临的挑战 | 本项目提出的解决方案 |
|:------------:|:----------:|
|   数据非独立同分布   |  数据增强框架方案  |
|    通信开销大     | ABY计算底层方案  |
|  客户端贡献度不均衡   |  客户端贡献度算法  |

<hr>

## 安装

- 请参考FATE官方文档
- 将本项目的federatedml目录替换FATE项目中的federatedml目录即可根据说明使用本项目的功能。


在已经成功部署FATE框架的基础上，激活FATE环境后，参考一下命令

```bash
$ cd $FATE_PROJECT_BASE/python
$ git clone https://github.com/sunset-under-forest/FATE_JzLab.git
$ cp -r FATE_JzLab/federatedml federatedml  # 替换原有federatedml目录
```
<hr>

## 使用示例

参照模块文档

<hr>

## 贡献

如果你希望其他人为项目做出贡献，提供贡献指南。

1. Fork 项目
2. 创建您的特性分支 (`git checkout -b feature/your-feature`)
3. 提交您的更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建一个新的拉取请求

<hr>

## 联系方式

Email:  **spongebob404@163.com**


<hr>

## 其他工作

**[FATE框架分析（点击此链接可查看分析文档）](test%2FFATE_ARCHITECTURE%2FREADME.md)**


&emsp;&emsp;作为本项目的开发者，我们从开始接手FATE框架的一开始，就认为，要在FATE框架中进行联邦学习算法开发，如果只是以FATE框架为舞台或是基底去提供解决方案，读完FATE框架提供的官方文档就开始着手写代码，测试的话，是不太可行的。要能够完整地实现我们提出的方案，加速开发进度，提高测试效率，我们必须要对FATE框架有一个全面的了解，不只是知道怎么可以用官方SDK开发，还要去分析FATE框架的代码，知道手上这个叫FATE的工具的运行原理，知道这个东西到底是怎么做出来的。
<br>
<br>

&emsp;&emsp;FATE框架更像是一个系统，高度封装，功能丰富，对开发者提供各种各样的接口，很类似于操作系统。就跟正常的编程一样，如果只是会一门编程语言，会基本的语法，通过基本接口去实现对抽象的实现，或许可以发挥出这门语言50%左右的能力，但是如果还了解底层，了解这门语言底层的实现逻辑，好比学了python去学c，学了c还去学操作系统，学了java去学JVM一样，那么可以发挥的就不只是这一门语言的能力了，手上的整台机器都能为自己所用。
<br>
<br>

&emsp;&emsp;一般来说，前者这种最多只能称是个理论学家，也并非说这样就是完全不可取的。我们是业内人士，当然不拘于仅仅提出一个方案然后运行通过就草草收工，**我们的目标是提出自己的创新方案，并且让我们的创意和想法不只是活在纸上，而是真正的实现出来，让我们的想法能够落地，能够真正地创造价值，推动技术的革新与进步**。
<br>
<br>

&emsp;&emsp;因此，我们从7月份开始，就开始了对FATE框架的分析，克服了对大型项目框架的恐惧，从最开始的拿到框架，不知道怎么入手去分析，到找到框架的入口点，一行代码一行代码逐层深入，再到对框架的各个模块进行分析，从设计模式去看每一个类，从框架架构去看每个模块之间的依赖……
<br>
<br>

&emsp;&emsp;其实这个过程看似充满艰难，但其实回报就我们开发者的角度来看，是值得的而甚至是成正比的，不知如何形容的就是它所带来的收益。我们认为很好的概括就是让我们从单文档，小脚本，学生水平的开发者，开始领略甚至开始进化到具有项目视野，全局思考能力的技术极客。数据库操作，网络编程，多线程，设计模式，框架架构，分布式系统……这些都是我们在分析FATE框架的过程中**重新**学习的知识，为什么是重新学习，因为其实这些内容在学校上课也好，日常自己涉猎也好都有接触过，或深或浅。而我们想表达的，不仅是这些分散的，可以各自成为一个领域或者学科的技术内容是研究FATE框架过程中收获的宝藏，更重要的是，FATE框架给我们带来了一个不一样的全局视野，去重新审视我们所学的零散的知识，让我们体会到了将分散的技术聚合在一起是什么样的，需要什么视角，需要什么样的思维方式。
<br>
<br>

&emsp;&emsp;很有意思的一个思想过程就是，在研究FATE框架的过程中，我们在不同阶段都产生了对不同技术职位的理解与憧憬，从**程序员**（开始接触框架代码），**测试工程师**（调试测试），**网络运维工程师**（部署环境，各种DEBUG），到**架构师**（领略到架构和设计模式，理解看似复杂其实蕴含大道理的封装代码），**系统工程师**（从模块甚至系统的视角去审视FATE框架），甚至是**产品经理**（从用户的角度去看待我们团队设计的产品，如何把好的想法表达出来，画各种流程图，用例图，时序图……），这些都是我们在研究FATE框架的过程中产生的思考。最后我个人非常想成为**CTO**（当然作为年轻人，万事皆需历练与成长方能独当一面，但是从此我已经成为了自己的CTO😀），也想从全局的角度尝试一下带领团队从技术上创造一些可以有价值的事物。
<br>
<br>

&emsp;&emsp;FATE框架带给我们团队的是一个实实在在项目的启蒙，比起在FATE中实现的一系列算法，在我们看来这个过程本身就是最宝贵的成果。






