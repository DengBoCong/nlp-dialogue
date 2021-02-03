import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class BahdanauAttention(nn.Module):
    """ bahdanau attention实现

    :param units: 隐层单元数
    """

    def __init__(self, units: int) -> None:
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(in_features=2 * units, out_features=units)
        self.W2 = nn.Linear(in_features=units, out_features=units)
        self.V = nn.Linear(in_features=units, out_features=1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param query: 隐层状态
        :param values: encoder输出状态
        """
        values = values.permute(1, 0, 2)
        hidden_with_time_axis = torch.unsqueeze(input=query, dim=1)
        score = self.V(torch.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        attention_weights = F.softmax(input=score, dim=1)
        context_vector = attention_weights * values
        context_vector = torch.sum(input=context_vector, dim=1)

        return context_vector, attention_weights
