import os
import json
import jieba
import numpy as np
import tensorflow as tf
from hlp.chat.common.utils import log_operator


def _check_file(raw_file: str, processed_file: str, remove_tokenized: bool = True):
    """
    对原始文本进行检查是否存在
    删除已存在的分词文本
    :param raw_file: 原始数据路径
    :param processed_file: 生成token数据保存路径
    :param remove_tokenized: 是否移除原有分词文本
    :return: 无返回值
    """
    if not os.path.exists(raw_file):
        print('数据集不存在： ', raw_file)
        exit(0)
    # 如果if_remove为True且已经分词的文件存在，要删除，因为后面的读写操作是边读边写
    if os.path.exists(processed_file) and remove_tokenized:
        os.remove(processed_file)


def to_single_turn_dataset(tokenized_data_path: str, qa_data_path: str, dict_path: str, vocab_size: int,
                           start_sign: str = "<start>", end_sign: str = "<end>", unk_sign: str = "<unk>",
                           max_data_size: int = 0, remove_tokenized: bool = True):
    """生成单轮对话数据集

    用于处理已经分词好的多轮次数据集的方法，将数据集处理成问答对的形式
    :param tokenized_data_path: 已切分多轮对话数据路径
    :param qa_data_path: 单轮对话数据保存路径
    :param dict_path: 字典保存路径
    :param vocab_size: 词汇量大小
    :param start_sign: 开始标记
    :param end_sign: 结束标记
    :param unk_sign: 未登录词
    :param max_data_size: 最大加载数据量，,0为所有数据
    :param remove_tokenized: 是否移除原有分词文本
    :return: 无返回值
    """
    # _check_file(raw_file=raw_data_path, processed_file=qa_data_path, remove_tokenized=remove_tokenized)

    count = 0
    sentences_count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []
    one_pair = []
    all_text_list = []

    # 对每一轮对话上下文进行配对，形成一问一答两个部分，如果遇到下一轮对话，直接跳过
    with open(tokenized_data_path, encoding="utf-8") as raw_file, \
            open(qa_data_path, 'w', encoding="utf-8") as single_turn_data_file:
        for line in raw_file:
            line = line.strip('\n').replace('/', '')
            # line = re.sub(r"[%s]+" % punctuation, "", line)
            # 因为原始数据集中，是一轮一轮的对话排列的，所以需要注意的是
            # 在一轮对话结束之后，最后一句不能作为问句，需要跳到下一轮进行处理
            if line == '':
                one_pair = []
                count += 1
                continue
            elif len(one_pair) == 1:
                one_pair.append(line)
                question = start_sign + " " + one_pair[0] + " " + end_sign
                answer = start_sign + " " + one_pair[1] + " " + end_sign
                single_turn_data_file.write(question + "\t" + answer + "\n")
                all_text_list.append(question)
                all_text_list.append(answer)
                one_pair = [line]
                sentences_count += 1
                print('\r已处理：{}个问答对'.format(sentences_count), flush=True, end="")
                if sentences_count == max_data_size:
                    break
            else:
                one_pair.append(line)

            length = len(line)
            max_len = max(max_len, length)
            min_len = min(min_len, length)
            sentence_len.append(length)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", num_words=vocab_size, oov_token=unk_sign)
    tokenizer.fit_on_texts(all_text_list)
    with open(dict_path, 'w', encoding='utf-8') as dict_file:
        dict_file.write(tokenizer.to_json())

    message = "对话数据集转换完毕，并保存字典：共处理{}轮对话数据，整理出{}对" \
              "问答对，语句最大长度：{}，语句最短长度{}，语句平均长度{:.3f}".format(count, sentences_count,
                                                           max_len, min_len, np.mean(sentence_len))
    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_raw_xiao_huang_ji_data(raw_data: str, tokenized_data: str, if_remove: bool = True):
    """
    用于处理小黄鸡数据集的方法，将小黄鸡数据集处理成多轮次对话的形式，并分词
    :param raw_data: 原始数据路径
    :param tokenized_data: 生成token数据保存路径
    :param if_remove: 是否移除原有分词文本
    :return:
    """
    _check_file(raw_file=raw_data, processed_file=tokenized_data, remove_tokenized=if_remove)

    count = 1
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_data, 'r', encoding="utf-8") as raw_file, open(tokenized_data, 'a',
                                                                 encoding="utf-8") as tokenized_file:
        for line in raw_file:
            line = line.strip('\n').replace('/', '')
            if line == "":
                tokenized_file.write("\n")
                count += 1
                print("\r已读取：{}轮对话数据".format(count), flush=True, end="")
                continue

            length = len(line)
            sentence_len.append(length)
            max_len = max(max_len, length)
            min_len = min(min_len, length)
            tokenized_file.write(" ".join(jieba.cut(line)) + "\n")

    message = "数据处理完毕，数据信息统计：共处理{}轮对话数据，语句最大长度：{}，语" \
              "句最短长度{}，语句平均长度{:.3f}".format(count, max_len, min_len, np.mean(sentence_len))

    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_raw_lccc_data(raw_data_path: str, tokenized_data_path: str, remove_tokenized: bool = True):
    """将LCCC数据集从JSON格式转换每行一条话语

    LCCC原始数据集已分词.

    :param raw_data_path: 原始数据路径
    :param tokenized_data_path: 生成token数据保存路径
    :param remove_tokenized: 是否移除原有分词文本
    :return: 无返回值
    """
    _check_file(raw_file=raw_data_path, processed_file=tokenized_data_path, remove_tokenized=remove_tokenized)

    count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_data_path, 'r', encoding="utf-8") as raw_file, open(tokenized_data_path, 'a',
                                                                      encoding="utf-8") as tokenized_file:
        raw_data_path = json.load(raw_file)
        for data in raw_data_path:
            for sentence in data:
                length = len(sentence)
                sentence_len.append(length)
                max_len = max(max_len, length)
                min_len = min(min_len, length)
                tokenized_file.write(sentence + "\n")

            tokenized_file.write("\n")
            count += 1

            print("\r已读取：{}轮对话数据".format(count), flush=True, end="")

    message = "数据预处理完毕：共处理{}轮对话数据，语句最大长度：{}，语" \
              "句最短长度{}，语句平均长度{:.3f}".format(count, max_len, min_len, np.mean(sentence_len))

    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_raw_douban_data(raw_data: str, tokenized_data: str, repeat_data: int = 10, if_remove: bool = True):
    """
    用于处理douban数据集的方法，将douban数据集处理成多轮次对话的形式，并分词
    :param raw_data: 原始数据路径
    :param tokenized_data: 生成token数据保存路径
    :param repeat_data: 每轮对话重复数据条数
    :param if_remove: 是否移除原有分词文本
    :return: 无返回值
    """
    _check_file(raw_file=raw_data, processed_file=tokenized_data, remove_tokenized=if_remove)

    count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_data, 'r', encoding='utf-8') as raw_file, open(tokenized_data, 'a',
                                                                 encoding='utf-8') as tokenized_file:
        iter_count = -1
        for line in raw_file:
            iter_count += 1
            if iter_count % repeat_data != 0:
                continue
            line = line.strip('\n').replace('/', '')
            if line == "":
                continue

            # 因为原始数据集中，是一轮一轮的对话排列的，所以需要注意的是在一轮对话结束之后，最后
            # 一句不能作为问句，需要跳到下一轮进行处理去掉最前面的标签和最后面的不正确语句
            utterances = line.split('\t')[1:-1]
            for utterance in utterances:
                length = len(utterance)
                sentence_len.append(length)
                max_len = max(max_len, length)
                min_len = min(min_len, length)
                tokenized_file.write(utterance + "\n")
            tokenized_file.write("\n")
            count += 1
            print("\r数据处理进度：{}".format(count), flush=True, end="")

    message = "数据处理完毕，数据信息统计：共处理{}轮对话数据，语句最大长度：{}，语" \
              "句最短长度{}，语句平均长度{:.3f}".format(count, max_len, min_len, np.mean(sentence_len))

    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_raw_cross_woz_data(raw_data: str, tokenized_data: str, if_remove: bool = True):
    """
    用于处理crossWOZ数据集的方法，将crossWOZ数据集处理成多轮次对话的形式，并分词
    :param raw_data: 原始数据路径
    :param tokenized_data: 生成token数据保存路径
    :param if_remove: 是否移除原有分词文本
    :return: 无返回值
    """
    _check_file(raw_file=raw_data, processed_file=tokenized_data, remove_tokenized=if_remove)

    count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_data, 'r', encoding='utf-8') as raw_file, open(tokenized_data, 'a',
                                                                 encoding='utf-8') as tokenized_file:
        raw_data = json.load(raw_file)
        for data in raw_data:
            turn_utterances = raw_data[data]["messages"]
            for content in turn_utterances:
                sentence = content["content"]
                length = len(sentence)
                sentence_len.append(length)
                max_len = max(max_len, length)
                min_len = min(min_len, length)
                tokenized_file.write(" ".join(jieba.cut(sentence)) + "\n")
            tokenized_file.write("\n")
            count += 1
            print("\r已读取：{}轮对话数据".format(count), flush=True, end="")

    message = "数据处理完毕，数据信息统计：共处理{}轮对话数据，语句最大长度：{}，语" \
              "句最短长度{}，语句平均长度{:.3f}".format(count, max_len, min_len, np.mean(sentence_len))

    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_raw_tie_ba_data(raw_data: str, tokenized_data: str, if_remove: bool = True):
    """
    用于处理TieBa数据集的方法，将TieBa数据集处理成多轮次对话的形式，并分词
    :param raw_data: 原始数据路径
    :param tokenized_data: 生成token数据保存路径
    :param if_remove: 是否移除原有分词文本
    :return: 无返回值
    """
    _check_file(raw_file=raw_data, processed_file=tokenized_data, remove_tokenized=if_remove)

    count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_data, 'r', encoding='utf-8') as raw_file, open(tokenized_data, 'a',
                                                                 encoding='utf-8') as tokenized_file:
        for line in raw_file:
            line = line.strip("\n").replace("/", " ")
            if line == '':
                continue

            line = line.split("\t")
            for sentence in line:
                length = len(sentence)
                sentence_len.append(length)
                max_len = max(max_len, length)
                min_len = min(min_len, length)
                tokenized_file.write(" ".join(jieba.cut(sentence)) + "\n")
            tokenized_file.write("\n")

            count += 1
            print("\r已读取：{}轮对话数据".format(count), flush=True, end="")

    message = "数据处理完毕，数据信息统计：共处理{}轮对话数据，语句最大长度：{}，语" \
              "句最短长度{}，语句平均长度{:.3f}".format(count, max_len, min_len, np.mean(sentence_len))

    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_raw_ppt_gossiping_data(raw_data: str, tokenized_data: str, if_remove: bool = True):
    """
    用于处理PPT-Gossiping数据集的方法，将PPT-Gossiping数据集处理成多轮次对话的形式，并分词
    :param raw_data: 原始数据路径
    :param tokenized_data: 生成token数据保存路径
    :param if_remove: 是否移除原有分词文本
    :return: 无返回值
    """
    # 由于原始数据格式和贴吧格式一致，直接调用贴吧数据处理方法
    preprocess_raw_tie_ba_data(raw_data, tokenized_data, if_remove=if_remove)


def preprocess_raw_wei_bo_data(raw_post_data: str, raw_response_data,
                               tokenized_data: str, if_remove: bool = True):
    """
    用于处理weibo数据集的方法，将weibo数据集处理成多轮次的形式，并分词
    :param raw_post_data: 微博的post原始文本数据中的路径
    :param raw_response_data: 微博的response原始文本数据中的路径
    :param tokenized_data: 生成token数据保存路径
    :param if_remove: 是否移除原有分词文本
    :return: 无返回值
    """
    _check_file(raw_file=raw_post_data, processed_file=tokenized_data, remove_tokenized=if_remove)
    if not os.path.exists(raw_response_data):
        print('数据集不存在，请添加数据集!')
        exit(0)

    count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_post_data, 'r', encoding='utf-8') as post_file, open(
            raw_response_data, 'r', encoding='utf-8') as response_file, \
            open(tokenized_data, 'a', encoding='utf-8') as tokenized_file:
        for post_data, response_data in zip(post_file, response_file):
            post_data = post_data.strip("\n").replace("/", " ")
            response_data = response_data.strip("\n").replace("/", " ")
            if post_data == "" or response_data == "":
                continue

            post_len = len(post_data)
            response_len = len(response_data)
            max_len = max(max_len, post_len, response_len)
            min_len = min(min_len, post_len, response_len)
            sentence_len.append(post_len)
            sentence_len.append(response_len)
            tokenized_file.write(post_data + "\n" + response_data + "\n\n")

            count += 1
            print("\r已读取：{}轮对话数据".format(count), flush=True, end="")

    message = "数据处理完毕：共处理{}轮对话数据，语句最大长度：{}，语" \
              "句最短长度{}，语句平均长度{:.3f}".format(count, max_len, min_len, np.mean(sentence_len))

    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_raw_qin_yun_data(raw_data: str, tokenized_data: str, if_remove: bool = True):
    """
    用于处理青云数据集的方法，将青云数据集处理成多轮次的形式，并分词
    :param raw_data: 原始数据路径
    :param tokenized_data: 生成token数据保存路径
    :param if_remove: 是否移除原有分词文本
    :return: 无返回值
    """
    _check_file(raw_file=raw_data, processed_file=tokenized_data, remove_tokenized=if_remove)

    count = 0
    max_len = 0
    min_len = 10000
    sentence_len = []

    with open(raw_data, 'r', encoding='utf-8') as raw_file, open(
            tokenized_data, 'a', encoding='utf-8') as tokenized_file:
        for line in raw_file:
            line = line.strip().strip("\n").replace("/", " ")
            if line == "":
                continue

            for sentence in line.split("|"):
                sentence = sentence.strip()

                length = len(sentence)
                sentence_len.append(length)
                max_len = max(max_len, length)
                min_len = min(min_len, length)
                tokenized_file.write(" ".join(jieba.cut(sentence)) + "\n")
            tokenized_file.write("\n")

            count += 1
            print("\r已读取：{}轮对话数据".format(count), flush=True, end="")

    message = "数据处理完毕，数据信息统计：共处理{}轮对话数据，语句最大长度：{}，语" \
              "句最短长度{}，语句平均长度{:.3f}".format(count, max_len, min_len, np.mean(sentence_len))

    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def combine_tokenized_data_single(standby_data: list, combine_data: str, if_remove: bool = True):
    """
    *单轮对话数据集处理模块*
    将所有已经分词好的问答对集中整合到一个文件中
    :param standby_data: 分词好的数据文本路径
    :param combine_data: 汇总数据的文本路径
    :param if_remove: 是否移除原有分词文本
    :return: 无返回值
    """
    if os.path.exists(combine_data) and if_remove:
        os.remove(combine_data)

    count = 0
    file_count = 0

    for file_fn in standby_data:
        if not os.path.exists(file_fn):
            print("{}文件不存在，请检查之后再次运行".format(file_fn))
            exit(0)
        with open(file_fn, 'r', encoding='utf-8') as tokenized_file, open(combine_data, 'a',
                                                                          encoding='utf-8') as combine_file:
            for line in tokenized_file:
                line = line.strip().strip("\n").replace("/", " ")
                combine_file.write(line + "\n")
                count += 1
                print("\r数据处理进度：{}".format(count), flush=True, end="")

        file_count += 1

    message = "数据处理完毕，数据信息统计：共处理{}个分词文件，整理出{}条数据".format(file_count, count)
    print("\n" + message)
    logger = log_operator(level=10)
    logger.info(message)


def preprocess_datasets(dataset_name: str, raw_data_path: str, tokenized_data_path: str,
                        remove_tokenized: bool = True, reserve_data: str = None):
    """对话数据集处理

    用来整合目前所有数据处理方法，通过字典匹配进行调用，默认使用preprocess_raw_lccc_data
    :param dataset_name: 对应分词方法的名称，作为key，目前有：xiaohuangji，tieba，ppt_gossiping，lccc，douban，cross_woz
    :param raw_data_path: 原始数据路径
    :param tokenized_data_path: 生成token数据保存路径
    :param remove_tokenized: 是否移除原有分词文本
    :param reserve_data: 原始文本备用参数
    :return: 无返回值
    """
    print("数据集：", dataset_name)
    operation = {
        "xiao_huang_ji": lambda: preprocess_raw_xiao_huang_ji_data(raw_data_path,
                                                                   tokenized_data_path, remove_tokenized),
        "tie_ba": lambda: preprocess_raw_tie_ba_data(raw_data_path, tokenized_data_path, remove_tokenized),
        "ppt_gossiping": lambda: preprocess_raw_ppt_gossiping_data(raw_data_path,
                                                                   tokenized_data_path, remove_tokenized),
        "lccc": lambda: preprocess_raw_lccc_data(raw_data_path, tokenized_data_path, remove_tokenized),
        "dou_ban": lambda: preprocess_raw_douban_data(raw_data_path, tokenized_data_path, 2, remove_tokenized),
        "cross_woz": lambda: preprocess_raw_cross_woz_data(raw_data_path, tokenized_data_path, remove_tokenized),
        "wei_bo": lambda: preprocess_raw_wei_bo_data(raw_data_path, reserve_data,
                                                     tokenized_data_path, remove_tokenized),
        "qin_yun": lambda: preprocess_raw_qin_yun_data(raw_data_path, tokenized_data_path, remove_tokenized)
    }

    operation.get(dataset_name, "lccc")()


def raw_to_tokenized_and_combine_single(standby_data: dict, combine_data: str, if_save_tokenized: bool = False):
    """
    *单轮对话数据集处理模块*
    提供一次性将所有原始数据文本转换成分词文件，并整合到一个文件中
    :param standby_data: 分词好的数据文本路径，分词方法匹配字典，key为对应的数据库名称，value为原始文本路径
                    目前提供的的方法有：{"xiao_huang_ji":"path","tie_ba":"path","ppt_gossiping":"path","lccc":"path",
                                        "dou_ban":"path","cross_woz":"path","wei_bo":"path","qin_yun":"path"}
    :param combine_data: 汇总数据的文本路径
    :param if_save_tokenized: 是否保留过程分词文件，如果为True，保留的各分词文件名直接在原始文件名后加tokenized，如lccc_tokenized.txt
    :return: 无返回值
    """
    tokenized_files = []
    for file in standby_data:
        print("正在处理{}语料".format(file))
        if if_save_tokenized:
            data_dir = "\\".join(standby_data[file].split("\\")[:-1])
            tokenized_dir = data_dir + "\\tokenized_data"
            tokenized_file = tokenized_dir + "\\" + file + "_tokenized.txt"
            if not os.path.exists(tokenized_dir):
                os.makedirs(tokenized_dir)
            preprocess_datasets(dataset_name=file, raw_data_path=standby_data[file],
                                tokenized_data_path=tokenized_file, remove_tokenized=True)
            tokenized_files.append(tokenized_file)
            print("已保存{}语料的分词文本".format(file))
        else:
            preprocess_datasets(dataset_name=file, raw_data_path=standby_data[file],
                                tokenized_data_path=combine_data, remove_tokenized=False)
            print("已合成{}语料".format(file))

    if if_save_tokenized:
        combine_tokenized_data_single(standby_data=tokenized_files, combine_data=combine_data)
    else:
        print("数据合成完毕，已保存至{}文件中，相关单文本信息已保存至日志文件中".format(combine_data))
