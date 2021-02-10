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
"""seq2seq的模型功能实现，包含train模式、evaluate模式、chat模式
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn.functional as F
import torch.utils.data as data
import random
from dialogue.pytorch.beamsearch import BeamSearch
from dialogue.pytorch.modules import Modules
from dialogue.tools import load_tokenizer
from dialogue.tools import preprocess_request
from dialogue.tools import ProgressBar
from torch.optim import Optimizer
from typing import AnyStr
from typing import Dict
from typing import NoReturn
from typing import Tuple


class Seq2SeqModules(Modules):
    def __init__(self, batch_size: int, max_sentence: int, train_data_type: str, valid_data_type: str,
                 dict_path: str = "", num_workers: int = 2, model: torch.nn.Module = None,
                 encoder: torch.nn.Module = None, decoder: torch.nn.Module = None,
                 device: torch.device = None) -> NoReturn:
        super(Seq2SeqModules, self).__init__(
            batch_size=batch_size, max_sentence=max_sentence, train_data_type=train_data_type, dict_path=dict_path,
            valid_data_type=valid_data_type, num_workers=num_workers, model=model, encoder=encoder, decoder=decoder,
            device=device
        )

    def _train_step(self, batch_dataset: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    optimizer: Optimizer, *args, **kwargs) -> Dict:
        """ 训练步

        :param batch_dataset: 训练步的当前batch数据
        :param optimizer: 优化器
        :return: 返回所得指标字典
        """
        inputs, targets, weights = \
            batch_dataset[0].to(torch.long), batch_dataset[1].to(torch.long), batch_dataset[2].to(torch.long)
        inputs = inputs.permute(1, 0)
        targets = targets.permute(1, 0)

        optimizer.zero_grad()
        enc_outputs, enc_state = self.encoder(inputs)
        dec_state = enc_state
        dec_input = targets[:1, :]
        outputs = torch.zeros(self.max_sentence, self.batch_size, kwargs["vocab_size"])
        for t in range(1, self.max_sentence):
            predictions, dec_hidden = self.decoder(dec_input, dec_state, enc_outputs)
            outputs[t] = predictions

            teacher_force = random.random() < kwargs["teacher_forcing_ratio"]
            top_first = torch.argmax(predictions, dim=-1)
            dec_input = (targets[t:t + 1] if teacher_force else top_first)

        outputs = torch.reshape(input=outputs[1:], shape=[-1, outputs.shape[-1]])
        targets = torch.reshape(input=targets[1:], shape=[-1])

        loss = torch.nn.CrossEntropyLoss(ignore_index=0)(outputs, targets)
        loss.backward()
        optimizer.step()

        return {"train_loss": loss}

    def _valid_step(self, loader: data.DataLoader, steps_per_epoch: int,
                    progress_bar: ProgressBar, *args, **kwargs) -> Dict:
        """ 验证模块

        :param loader: 数据加载器
        :param steps_per_epoch: 验证步总步数
        :param progress_bar: 进度控制器
        :return: 返回所得指标字典
        """
        print("验证轮次")
        total_loss = 0
        start_time = time.time()
        progress_bar = ProgressBar(total=steps_per_epoch, num=self.batch_size)

        with torch.no_grad():
            for (batch, (inputs, targets, _)) in enumerate(loader):
                inputs, targets = inputs.to(torch.long), targets.to(torch.long)
                inputs, targets = inputs.permute(1, 0), targets.permute(1, 0)
                inputs, targets = inputs.to(kwargs["device"]), targets.to(kwargs["device"])

                enc_outputs, enc_state = self.encoder(inputs)
                dec_state = enc_state
                dec_input = targets[:1, :]
                outputs = torch.zeros(self.max_sentence, self.batch_size, kwargs["vocab_size"]).to(kwargs["device"])
                for t in range(1, self.max_sentence):
                    predictions, dec_hidden = self.decoder(dec_input, dec_state, enc_outputs)
                    outputs[t] = predictions
                    dec_input = torch.argmax(predictions, dim=-1)

                outputs = torch.reshape(input=outputs[1:], shape=[-1, outputs.shape[-1]])
                targets = torch.reshape(input=targets[1:], shape=[-1])
                loss = torch.nn.CrossEntropyLoss(ignore_index=0)(outputs, targets)
                total_loss += loss

                progress_bar(current=batch + 1, metrics="- valid_loss: {:.4f}".format(loss))

        progress_bar.done(step_time=time.time() - start_time)

        return {"valid_loss": total_loss / steps_per_epoch}

    def inference(self, request: str, beam_size: int, start_sign: str = "<start>", end_sign: str = "<end>") -> AnyStr:
        """ 对话推断模块

        :param request: 输入句子
        :param beam_size: beam大小
        :param start_sign: 句子开始标记
        :param end_sign: 句子结束标记
        :return: 返回历史指标数据
        """
        with torch.no_grad():
            tokenizer = load_tokenizer(self.dict_path)
            enc_input = preprocess_request(sentence=request, tokenizer=tokenizer,
                                           max_length=self.max_sentence, start_sign=start_sign, end_sign=end_sign)
            enc_input = torch.tensor(data=enc_input, dtype=torch.long).permute(1, 0)
            enc_output, states = self.encoder(inputs=enc_input)
            dec_input = torch.tensor(data=[[tokenizer.word_index.get(start_sign)]])

            beam_search_container = BeamSearch(beam_size=beam_size, max_length=self.max_sentence, worst_score=0)
            beam_search_container.reset(enc_output=enc_output.permute(1, 0, 2), dec_input=dec_input, remain=states)
            enc_output, dec_input, states = beam_search_container.get_search_inputs()
            enc_output = enc_output.permute(1, 0, 2)

            for t in range(self.max_sentence):
                predictions, dec_hidden = self.decoder(dec_input, states, enc_output)
                predictions = F.softmax(input=predictions, dim=-1)

                beam_search_container.expand(predictions=predictions[0], end_sign=tokenizer.word_index.get(end_sign))
                if beam_search_container.beam_size == 0:
                    break

                enc_output, dec_input, states = beam_search_container.get_search_inputs()
                dec_input = dec_input[:, -1].unsqueeze(-1)
                enc_output = enc_output.permute(1, 0, 2)
                dec_input = dec_input.permute(1, 0)

            beam_search_result = beam_search_container.get_result(top_k=3)
            result = ""
            # 从容器中抽取序列，生成最终结果
            for i in range(len(beam_search_result)):
                temp = beam_search_result[i].numpy()
                text = tokenizer.sequences_to_texts(temp)
                text[0] = text[0].replace(start_sign, "").replace(end_sign, "").replace(" ", "")
                result = "<" + text[0] + ">" + result
            return result
