<h1 align="center">NLP-Dialogue</h1>

<div align="center">

[![Blog](https://img.shields.io/badge/blog-@DengBoCong-blue.svg?style=social)](https://www.zhihu.com/people/dengbocong)
[![Paper Support](https://img.shields.io/badge/paper-repo-blue.svg?style=social)](https://github.com/DengBoCong/nlp-paper)
![Stars Thanks](https://img.shields.io/badge/Stars-thanks-brightgreen.svg?style=social&logo=trustpilot)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=social&logo=appveyor)

[comment]: <> ([![PRs Welcome]&#40;https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square&#41;]&#40;&#41;)

</div>

<h3 align="center">[English](https://github.com/DengBoCong/nlp-dialogue) | [中文](https://github.com/DengBoCong/nlp-dialogue/blob/main/README.CN.md) </h3>

# 架构
开放域生成问答模型
Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering

检索
Dense Passage Retrieval for Open-Domain Question Answering

# 工具
APScheduler

# 项目正在优化架构，可执行代码已经标记tag

一个能够部署执行的全流程对话系统
+ TensorFlow模型
   + Transformer
   + Seq2Seq
   + SMN检索式模型
   + Scheduled Sampling的Transformer
   + GPT2
   + Task Dialogue
+ Pytorch模型
   + Transformer
   + Seq2Seq

# 项目说明

本项目奔着构建一个能够在线部署对话系统，同时包含开放域和面向任务型两种对话系统，针对相关模型进行复现，论文阅读笔记放置另一个项目：[nlp-paper](https://github.com/DengBoCong/nlp-paper)，项目中使用TensorFlow和Pytorch进行实现。

# 语料
仓库中的[data](https://github.com/DengBoCong/nlp-dialogue/tree/main/dialogue/data)目录下放着各语料的玩具数据，可用于验证系统执行性，完整语料以及Paper可以在[这里](https://github.com/DengBoCong/nlp-paper)查看

+ LCCC
+ CrossWOZ
+ 小黄鸡
+ 豆瓣
+ Ubuntu
+ 微博
+ 青云
+ 贴吧

# 执行说明

+ Linux执行run.sh，项目工程目录检查执行check.sh（或check.py）
+ 根目录下的actuator.py为总执行入口，通过调用如下指令格式执行（执行前注意安装requirements.txt）：
```
python actuator.py --version [Options] --model [Options] ...
```
+ 通过根目录下的actuator.py进行执行时，`--version`、`--model`和`--act`为必传参数，其中`--version`为代码版本`tf/torch`，`--model`为执行对应的模型`transformer/smn...`，而act为执行模式（缺省状态下为`pre_treat`模式），更详细指令参数参见各模型下的`actuator.py`或config目录下的对应json配置文件。
+ `--act`执行模式说明如下：
   + pre_treat模式为文本预处理模式，如果在没有分词结果集以及字典的情况下，需要先运行pre_treat模式
   + train模式为训练模式
   + evaluate模式为指标评估模式
   + chat模式为对话模式，chat模式下运行时，输入ESC即退出对话。
+ 正常执行顺序为pre_treat->train->evaluate->chat
+ 各模型下单独有一个actuator.py，可以绕开外层耦合进行执行开发，不过执行时注意调整工程目录路径

# 目录结构说明
+ dialogue下为相关模型的核心代码放置位置，方便日后进行封装打包等
   + checkpoints为检查点保存位置
   + config为配置文件保存目录
   + data为原始数据储存位置，同时，在模型执行过程中产生的中间数据文件也保存在此目录下
   + models为模型保存目录
   + tensorflow及pytorch放置模型构建以及各模组执行的核心代码
   + preprocess_corpus.py为语料处理脚本，对各语料进行单轮和多轮对话的处理，并规范统一接口调用
   + read_data.py用于load_dataset.py的数据加载格式调用
   + metrics.py为各项指标脚本
   + tools.py为工具脚本，保存有分词器、日志操作、检查点保存/加载脚本等
+ docs下放置文档说明，包括模型论文阅读笔记
+ docker（mobile）用于服务端（移动终端）部署脚本
+ server为UI服务界面，使用flask进行构建使用，执行对应的server.py即可
+ tools为预留工具目录
+ actuator.py（run.sh）为总执行器入口
+ check.py（check.sh）为工程目录检查脚本


# SMN模型运行说明
SMN检索式对话系统使用前需要准备solr环境，solr部署系统环境推荐Linux，工具推荐使用容器部署(推荐Docker)，并准备：
+ Solr(8.6.3)
+ pysolr(3.9.0)

以下提供简要说明，更详细可参见文章：[搞定检索式对话系统的候选response检索--使用pysolr调用Solr](https://zhuanlan.zhihu.com/p/300165220)
## Solr环境
需要保证solr在线上运行稳定，以及方便后续维护，请使用DockerFile进行部署，DockerFile获取地址：[docker-solr](https://github.com/docker-solr/docker-solr)

仅测试模型使用，可使用如下最简构建指令：
```
docker pull solr:8.6.3
# 然后启动solr
docker run -itd --name solr -p 8983:8983 solr:8.6.3
# 然后创建core核心选择器，这里取名smn(可选)
docker exec -it --user=solr solr bin/solr create_core -c smn
```

关于solr中分词工具有IK Analyzer、Smartcn、拼音分词器等等，需要下载对应jar，然后在Solr核心配置文件managed-schema中添加配置。

**特别说明**：如果使用TF-IDF，还需要在managed-schema中开启相似度配置。
## Python中使用说明
线上部署好Solr之后，在Python中使用pysolr进行连接使用：
```
pip install pysolr
```

添加索引数据（一般需要先安全检查）方式如下。将回复数据添加索引，responses是一个json,形式如：[{},{},{},...]，里面每个对象构建按照你回复的需求即可：
```
solr = pysolr.Solr(url=solr_server, always_commit=True, timeout=10)
# 安全检查
solr.ping()
solr.add(docs=responses)
```

查询方式如下，以TF-IDF查询所有语句query语句方式如下：
```
{!func}sum(product(idf(utterance,key1),tf(utterance,key1),product(idf(utterance,key2),tf(utterance,key2),...)
```

使用前需要先将数据添加至Solr，在本SMN模型中使用，先执行pre_treat模式即可。

# Demo概览
<img align="center" height="400" src="https://github.com/DengBoCong/nlp-dialogue/blob/main/assets/main.png">
<img align="center" height="400" src="https://github.com/DengBoCong/nlp-dialogue/blob/main/assets/chat.png">

# 参考代码和文献
1. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/250946855)：Transformer的开山之作，值得精读 | Ashish et al,2017
2. [Sequential Matching Network: A New Architecture for Multi-turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1612.01627v2.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/270554147)：SMN检索式对话模型，多层多粒度提取信息 | Devlin et al,2018
3. [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/328801239)：展示了以NMT架构超参数为例的首次大规模分析，实验为构建和扩展NMT体系结构带来了新颖的见解和实用建议。 | Denny et al,2017
4. [Scheduled Sampling for Transformers](https://arxiv.org/pdf/1906.07651.pdf) | [阅读笔记](https://zhuanlan.zhihu.com/p/267146739)：在Transformer应用Scheduled Sampling | Mihaylova et al,2019

# License
Licensed under the Apache License, Version 2.0. Copyright 2021 DengBoCong. [Copy of the license]().