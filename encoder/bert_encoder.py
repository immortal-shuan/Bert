import torch.nn as nn
from layers.transform_layer import TransformencoderLayer

"""
BERT encoder exploits 12 or 24 transformer layers to extract features.
"""


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()

        self.layers_num = args.layers_num

        self.linear = nn.Linear(args.emb_size, args.hidden_size)
        self.transformer = nn.ModuleList(
            [TransformencoderLayer(args) for _ in range(args.layers_num)])

    def forward(self, emb, mask):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        hidden = self.linear(emb)
        seq_length = hidden.size(1)

        # mask: [batch_size x 1 x seq_length x seq_length]
        mask = (mask > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1).float()
        mask = (1.0 - mask) * -10000.0

        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden
