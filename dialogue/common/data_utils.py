import os
import jieba
import pysolr
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from hlp.utils.utils import load_tokenizer


def preprocess_request(sentence: str, max_length: int, start_sign: str,
                       tokenizer: tf.keras.preprocessing.text.Tokenizer):
    """
    用于处理回复功能的输入句子，返回模型使用的序列
    :param sentence: 待处理句子
    :param max_length: 单个句子最大长度
    :param start_sign: 开始标记
    :param tokenizer: 分词器
    :return: 处理好的句子和decoder输入
    """
    sentence = " ".join(jieba.cut(sentence))

    inputs = tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=max_length, padding='post')
    dec_input = tf.expand_dims([tokenizer.word_index.get(start_sign)], 0)

    return inputs, dec_input


def _create_dataset(data_path: str, num_examples: int):
    """
    用于将分词文本读入内存，并整理成问答对
    :param data_path: 分词文本路径
    :param num_examples: 读取的数据量大小
    :return: 整理好的问答对和样本权重
    """
    if not os.path.exists(data_path):
        print('不存在已经分词好的文件，请先执行pre_treat模式')
        exit(0)

    with open(data_path, 'r', encoding="utf-8") as file:
        lines = file.read().strip().split('\n')
        sample_weights = []
        qa_pairs = []
        if num_examples != 0:
            lines = lines[:num_examples]

        for line in lines:
            # 文本数据中的问答对权重通过在问答对尾部添加“<|>”配置
            temp = line.split("<|>")
            qa_pairs.append([sentence for sentence in temp[0].split('\t')])
            # 如果没有配置对应问答对权重，则默认为1.
            if len(temp) == 1:
                sample_weights.append(float(1))
            else:
                sample_weights.append(float(temp[1]))

    return zip(*qa_pairs), sample_weights


def _read_data(data_path: str, num_examples: int, max_length: int, tokenizer: tf.keras.preprocessing.text.Tokenizer):
    """
    读取数据，将input和target进行分词后返回
    :param data_path: 分词文本路径
    :param num_examples: 读取的数据量大小
    :param max_length: 最大序列长度
    :param tokenizer: 传入现有的分词器，默认重新生成
    :return: 输入序列张量、目标序列张量和分词器
    """
    (input_lang, target_lang), diag_weight = _create_dataset(data_path, num_examples)
    input_tensor = tokenizer.texts_to_sequences(input_lang)
    target_tensor = tokenizer.texts_to_sequences(target_lang)

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length,
                                                                  padding='post')

    return input_tensor, target_tensor, diag_weight


def load_data(dict_fn: str, data_fn: str, buffer_size: int, batch_size: int, checkpoint_dir: str,
              max_length: int, valid_data_split: float = 0.0, valid_data_fn: str = "",
              max_train_data_size: int = 0, max_valid_data_size: int = 0):
    """
    数据加载方法，含四个元素的元组，包括如下：
    :param dict_fn: 字典路径
    :param data_fn: 文本数据路径
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param checkpoint_dir: 检查点保存路径
    :param max_length: 单个句子最大长度
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param valid_data_fn: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :return: 训练Dataset、验证Dataset、训练数据总共的步数、验证数据总共的步数和检查点前缀
    """
    print("读取训练对话对...")
    tokenizer = load_tokenizer(dict_path=dict_fn)
    train_input, train_target, sample_weights = \
        _read_data(data_path=data_fn, num_examples=max_train_data_size, max_length=max_length, tokenizer=tokenizer)

    valid_flag = True  # 是否开启验证标记
    valid_steps_per_epoch = 0

    if valid_data_fn != "":
        print("读取验证对话对...")
        valid_input, valid_target, _ = _read_data(data_path=valid_data_fn, num_examples=max_valid_data_size,
                                                  max_length=max_length, tokenizer=tokenizer)
    elif valid_data_split != 0.0:
        train_size = int(len(train_input) * (1.0 - valid_data_split))
        valid_input = train_input[train_size:]
        valid_target = train_target[train_size:]
        train_input = train_input[:train_size]
        train_target = train_target[:train_size]
        sample_weights = sample_weights[:train_size]
    else:
        valid_flag = False

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_target, sample_weights)).cache().shuffle(
        buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)

    if valid_flag:
        valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input, valid_target)).cache().shuffle(
            buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = valid_dataset.batch(batch_size, drop_remainder=True)
        valid_steps_per_epoch = len(valid_input) // batch_size
    else:
        valid_dataset = None

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    steps_per_epoch = len(train_input) // batch_size

    return train_dataset, valid_dataset, steps_per_epoch, valid_steps_per_epoch, checkpoint_prefix


def smn_load_train_data(dict_fn: str, data_fn: str, checkpoint_dir: str, buffer_size: int,
                        batch_size: int, max_utterance: int, max_sentence: int, max_train_data_size: int = 0):
    """
    用于SMN的训练数据加载
    :param dict_fn: 字典文本路径
    :param data_fn: 数据文本路径
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param checkpoint_dir: 检查点保存路径
    :param max_utterance: 每轮对话最大对话数
    :param max_sentence: 单个句子最大长度
    :param max_train_data_size: 最大训练数据量
    :return: TensorFlow的数据处理类、分词器、检查点前缀和总的步数
    """
    if not os.path.exists(data_fn):
        print('不存在训练数据集，请添加数据集之后重试')
        exit(0)

    print('正在读取文本数据...')
    history = []  # 用于保存每轮对话历史语句
    response = []  # 用于保存每轮对话的回答
    label = []  # 用于保存每轮对话的标签
    count = 0  # 用于处理数据计数

    with open(data_fn, 'r', encoding='utf-8') as file:
        odd_flag = True
        for line in file:
            odd_flag = not odd_flag
            if odd_flag:
                continue

            count += 1
            apart = line.split('\t')
            label.append(int(apart[0]))
            response.append(apart[-1])
            del apart[0]
            del apart[-1]
            history.append(apart)

            print('\r已读取 {} 轮对话'.format(count), flush=True, end="")
            if max_train_data_size == count:
                break

    tokenizer = load_tokenizer(dict_path=dict_fn)
    response = tokenizer.texts_to_sequences(response)
    response = tf.keras.preprocessing.sequence.pad_sequences(response, maxlen=max_sentence, padding="post")

    count = 0
    utterances = []
    for utterance in history:
        count += 1
        pad_sequences = [0] * max_sentence
        # 注意了，这边要取每轮对话的最后max_utterances数量的语句
        utterance_padding = tokenizer.texts_to_sequences(utterance)[-max_utterance:]
        utterance_len = len(utterance_padding)
        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != 10:
            utterance_padding += [pad_sequences] * (max_utterance - utterance_len)
        utterances.append(tf.keras.preprocessing.sequence.pad_sequences(utterance_padding, maxlen=max_sentence,
                                                                        padding="post").tolist())
        print('\r已生成 {} 轮训练数据'.format(count), flush=True, end="")

    print('数据生成完毕，正在转换为Dataset')
    dataset = tf.data.Dataset.from_tensor_slices((utterances, response, label)).cache().shuffle(
        buffer_size).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    steps_per_epoch = len(utterances) // batch_size
    print('训练数据处理完成，正在进行训练')

    return dataset, tokenizer, checkpoint_prefix, steps_per_epoch


def load_smn_valid_data(data_fn: str, max_sentence: int, max_utterance: int, max_valid_data_size: int,
                        tokenizer: tf.keras.preprocessing.text.Tokenizer = None,
                        max_turn_utterances_num: int = 10):
    """
    用于单独加载smn的评价数据，这个方法设计用于能够同时在train时进行评价，以及单独evaluate模式中使用
    注意了，这里token_dict和必传其一，同时传只使用tokenizer
    :param data_fn: 评价数据地址
    :param max_sentence: 最大句子长度
    :param max_utterance: 最大轮次语句数量
    :param max_valid_data_size: 最大验证数据量
    :param tokenizer: 分词器实例
    :param max_turn_utterances_num: dataset的批量，最好取单轮对话正负样本数总和的倍数
    :return: dataset
    """
    if not os.path.exists(data_fn):
        print('不存在验证数据集，请添加数据集之后重试')
        return

    history = []
    response = []
    label = []
    with open(data_fn, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split("\n")[:max_valid_data_size]
        for line in lines:
            apart = line.split("\t")
            label.append(int(apart[0]))
            response.append(apart[-1])
            del apart[0]
            del apart[-1]
            history.append(apart)

    response = tokenizer.texts_to_sequences(response)
    response = tf.keras.preprocessing.sequence.pad_sequences(response, maxlen=max_sentence, padding="post")

    utterances = []
    for utterance in history:
        pad_sequences = [0] * max_sentence
        utterance_padding = tokenizer.texts_to_sequences(utterance)[-max_utterance:]

        utterance_len = len(utterance_padding)
        # 如果当前轮次中的历史语句不足max_utterances数量，需要在尾部进行填充
        if utterance_len != max_utterance:
            utterance_padding += [pad_sequences] * (max_utterance - utterance_len)
        utterances.append(tf.keras.preprocessing.sequence.pad_sequences(utterance_padding, maxlen=max_sentence,
                                                                        padding="post").tolist())

    # 在这里不对数据集进行打乱，方便用于指标计算
    dataset = tf.data.Dataset.from_tensor_slices((utterances, response, label)).prefetch(
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(max_turn_utterances_num, drop_remainder=True)

    return dataset


def get_tf_idf_top_k(history: list, k: int = 5):
    """
    使用tf_idf算法计算权重最高的k个词，并返回
    :param history: 上下文语句
    :param k: 返回词数量
    :return: top_5_key
    """
    tf_idf = {}

    vectorizer = TfidfVectorizer(analyzer='word')
    weights = vectorizer.fit_transform(history).toarray()[-1]
    key_words = vectorizer.get_feature_names()

    for i in range(len(weights)):
        tf_idf[key_words[i]] = weights[i]

    top_k_key = []
    tf_idf_sorted = sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:k]
    for element in tf_idf_sorted:
        top_k_key.append(element[0])

    return top_k_key


def creat_index_dataset(data_fn: str, solr_sever: str, max_database_size: int,
                        vocab_size: int, dict_path: str, unk_sign: str = "<unk>"):
    """
    生成轮次tf-idf为索引的候选回复
    :param data_fn: 文本数据路径
    :param solr_sever: solr服务的地址
    :param max_database_size: 从文本中读取最大数据量
    :param dict_path: 字典保存路径
    :param vocab_size: 词汇量大小
    :param unk_sign: 未登录词
    :return: 无返回值
    """
    if not os.path.exists(data_fn):
        print("没有找到对应的文本数据，请确认文本数据存在")
        exit(0)

    responses = []
    count = 0
    all_text_list = []
    solr = pysolr.Solr(url=solr_sever, always_commit=True)
    solr.ping()

    print("检测到对应文本，正在处理文本数据")
    with open(data_fn, 'r', encoding='utf-8') as file:
        odd_flag = True
        for line in file:
            odd_flag = not odd_flag
            if odd_flag:
                continue

            count += 1
            line = line.strip('\n').replace('/', '')
            apart = line.split("\t")[1:]
            all_text_list.extend(apart)
            for i in range(len(apart)):
                responses.append({"utterance": apart[i]})

            print("\r已处理了 {} 轮次对话".format(count), flush=True, end="")
            if max_database_size == count:
                break

    solr.delete(q="*:*")
    solr.add(docs=responses)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", num_words=vocab_size, oov_token=unk_sign)
    tokenizer.fit_on_texts(all_text_list)
    with open(dict_path, 'w', encoding='utf-8') as dict_file:
        dict_file.write(tokenizer.to_json())

    print("\n文本处理完毕，已更新候选回复集，并且以保存字典")
