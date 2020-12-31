import math
import torch
import torch.nn as nn


class MultiHeadedAtt(nn.Module):
    def __init__(self, args):
        super(MultiHeadedAtt, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads_num = args.heads_num
        self.heads_size = self.hidden_size // self.heads_num

        self.k_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.q_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, key, query, value, mask):
        """
        input:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """

        batch_size, seq_length, hidden_size = query.size()

        query = self.q_linear(query).view(batch_size, -1, self.heads_num, self.heads_size)
        key = self.k_linear(key).view(batch_size, -1, self.heads_num, self.heads_size)
        value = self.v_linear(value).view(batch_size, -1, self.heads_num, self.heads_size)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.heads_size))
        probs = self.dropout(self.softmax(scores + mask))

        output = self.linear(torch.matmul(probs, value).view(batch_size, -1, self.hidden_size))
        return output
