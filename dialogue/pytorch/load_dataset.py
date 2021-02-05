# Copyright 2021 DengBoCong. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Dataset加载模块，内含各模型针对性的以及公用性的数据加载方法
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dialogue.tools import load_tokenizer

def load_data(dict_path: str, buffer_size: int, batch_size: int, train_data_type: str, valid_data_type: str,
              max_sentence: int, valid_data_split: float = 0.0, train_data_path: str = "", valid_data_path: str = "",
              max_train_data_size: int = 0, max_valid_data_size: int = 0, **kwargs):
    """ 数据加载方法

    :param dict_path: 字典路径
    :param buffer_size: Dataset加载缓存大小
    :param batch_size: Dataset加载批大小
    :param train_data_type: 读取训练数据类型，单轮/多轮...
    :param valid_data_type: 读取验证数据类型，单轮/多轮...
    :param max_sentence: 单个句子最大长度
    :param valid_data_split: 用于从训练数据中划分验证数据
    :param train_data_path: 文本数据路径
    :param valid_data_path: 验证数据文本路径
    :param max_train_data_size: 最大训练数据量
    :param max_valid_data_size: 最大验证数据量
    :return: 训练Dataset、验证Dataset、训练数据总共的步数、验证数据总共的步数和检查点前缀
    """
    tokenizer = load_tokenizer(dict_path=dict_path)

    train_flag = True  # 是否开启训练标记
    train_steps_per_epoch = 0
    train_first, train_second, train_third = None, None, None

    valid_flag = True  # 是否开启验证标记
    valid_steps_per_epoch = 0
    valid_first, valid_second, valid_third = None, None, None

    if train_data_path != "":
        train_first, train_second, train_third = _read_data(
            data_path=train_data_path, max_data_size=max_train_data_size,
            max_sentence=max_sentence, data_type=train_data_type, tokenizer=tokenizer, **kwargs
        )
    else:
        train_flag = False

    print("训练数据读取中...")
    (input_lang, target_lang), diag_weight = read_tokenized_data(train_data_path, start_sign, end_sign, max_train_data_size)
    diag_weight = torch.tensor(diag_weight, dtype=torch.float32)
    # 合并input，target用于生成统一的字典
    lang = np.hstack((input_lang, target_lang))
    print("读取完成，正在格式化训练数据...")
    tokenizer = StaticTokenizerEncoder(sample=lang, tokenize=lambda x: x.split())
    # 将文本序列转换文token id之后，并进行填充
    input_data = [pad_tensor(tensor=tokenizer.encode(example)[:max_length], length=max_length, padding_index=0) for
                  example in input_lang]
    target_data = [pad_tensor(tensor=tokenizer.encode(example)[:max_length], length=max_length, padding_index=0) for
                   example in target_lang]
    input_tensor = stack_and_pad_tensors(input_data)[0]
    target_tensor = stack_and_pad_tensors(target_data)[0]

    print("格式化完成，正在整理训练数据并保存字典")
    word_index = {}
    vocab_list = tokenizer.vocab
    for i in range(tokenizer.vocab_size):
        word_index[vocab_list[i]] = i
        word_index[i] = vocab_list[i]

    with open(dict_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(word_index, indent=4, ensure_ascii=False))
    print("数据字典保存完成！")

    dataset = PairDataset(input_tensor, target_tensor, diag_weight)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    steps_per_epoch = len(input_tensor) // batch_size

    return loader, steps_per_epoch


