
# 运行说明
+ 运行入口：
   + seq2seq_chatter.py为seq2seq的执行入口文件：指令需要附带运行参数
   + transformer_chatter.py为transformer的执行入口文件：指令需要附带运行参数
   + smn_chatter.py为smn的执行入口文件：指令需要附带运行参数
+ 执行的指令格式：
   + seq2seq：python seq2seq_chatter.py --act [执行模式]
   + transformer：python transformer_chatter.py --act [执行模式]
   + smn：python smn_chatter.py --act [执行模式/pre_treat/train/evaluate/chat]
+ 执行类别：pre_treat(默认)/train/chat
+ 执行指令示例：
   + python seq2seq_chatter.py
   + python seq2seq_chatter.py --act pre_treat
   + python transformer_chatter.py
   + python transformer_chatter.py --act pre_treat
   + python smn_chatter.py
   + python smn_chatter.py --act pre_treat
+ pre_treat模式为文本预处理模式，如果在没有分词结果集的情况下，需要先运行pre_treat模式
+ train模式为训练模式
+ evaluate模式为指标评估模式
+ chat模式为对话模式。chat模式下运行时，输入exit即退出对话。

+ 正常执行顺序为pre_treat->train->evaluate->chat

# SMN模型运行说明
SMN检索式对话系统使用前需要准备solr环境，solr部署系统环境推荐Linux，工具推荐使用容器部署(推荐Docker)，并准备：
+ Solr(8.6.3)
+ pysolr(3.9.0)
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
