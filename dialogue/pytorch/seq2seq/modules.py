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
import random
from dialogue.pytorch.modules import Modules
from dialogue.tools import ProgressBar
from torch.optim import Optimizer
from torch.utils.data import DataLoader
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

    def _valid_step(self, loader: DataLoader, steps_per_epoch: int,
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

        for (batch, (inputs, targets, _)) in enumerate(loader):
            inputs = inputs.permute(1, 0)
            targets = targets.permute(1, 0)

            enc_outputs, enc_state = self.encoder(inputs)
            dec_state = enc_state
            dec_input = targets[:1, :]
            outputs = torch.zeros(self.max_sentence, self.batch_size, kwargs["vocab_size"])
            for t in range(1, self.max_sentence):
                predictions, dec_hidden = self.decoder(dec_input, dec_state, enc_outputs)
                outputs[t] = predictions
                dec_input = torch.argmax(predictions, dim=-1)

            outputs = torch.reshape(outputs[1:], shape=[-1, outputs.shape[-1]])
            targets = torch.reshape(targets[1:], shape=[-1])
            loss = torch.nn.CrossEntropyLoss(ignore_index=0)(outputs, targets)
            total_loss += loss

            progress_bar(current=batch + 1, metrics="- train_loss: {:.4f}"
                         .format(loss))

        progress_bar.done(step_time=time.time() - start_time)

        return {"valid_loss": total_loss / steps_per_epoch}

    def inference(self, *args, **kwargs) -> AnyStr:
        return None
