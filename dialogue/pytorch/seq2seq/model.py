import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """ seq2seq的encoder """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int, cell_type: str,
                 dec_units: int, num_layers: int, dropout: float, if_bidirectional: bool = True) -> None:
        """
        :param vocab_size: 词汇量大小
        :param embedding_dim: 词嵌入维度
        :param enc_units: encoder单元大小
        :param cell_type: cell类型，lstm/gru， 默认lstm
        :param dec_units: decoder单元大小
        :param num_layers: encoder中内部RNN层数
        :param dropout: 采样率
        :param if_bidirectional: 是否双向
        :return: Seq2Seq的Encoder
        """
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = list()

        if cell_type == "lstm":
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=enc_units,
                               num_layers=num_layers, bidirectional=if_bidirectional)
        elif cell_type == "gru":
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=enc_units,
                              num_layers=num_layers, bidirectional=if_bidirectional)

        self.fc = nn.Linear(in_features=enc_units * 2, out_features=dec_units)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        inputs = self.embedding(inputs)
        dropout = self.dropout(inputs)
        outputs, state = self.rnn(dropout)
        # 这里使用了双向GRU，所以这里将两个方向的特征层合并起来，维度将会是units * 2
        state = torch.tanh(self.fc(torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)))
        return outputs, state





class Decoder(nn.Module):
    """
    seq2seq的decoder，将初始化的inputs、隐藏层和encoder的输出作为
    输入，encoder的输入用来和隐藏层进行attention，得到的上下文
    向量和x进行整合然后丢到gru里去，最后Linear输出一下
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int,
                 dec_units: int, dropout: float, attention: nn.Module):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.gru = nn.GRU(input_size=(enc_units * 2) +
                                     embedding_dim, hidden_size=dec_units)
        self.fc = nn.Linear(in_features=(enc_units * 3) +
                                        embedding_dim, out_features=vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor, enc_output: torch.Tensor) -> Tuple[torch.Tensor]:
        inputs = inputs.unsqueeze(0)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        embedding = self.dropout(self.embedding(inputs))
        gru_inputs = torch.cat(
            (embedding, torch.unsqueeze(context_vector, dim=0)), dim=-1)
        output, dec_state = self.gru(gru_inputs, hidden.unsqueeze(0))
        embedding = embedding.squeeze(0)
        output = output.squeeze(0)
        context_vector = context_vector
        output = self.fc(
            torch.cat((embedding, context_vector, output), dim=-1))

        return output, dec_state.squeeze(0)
