import torch.nn as nn
from layers.embeding import WordPosSegEmbedding
from layers.MultiHeadedAttention import MultiHeadedAtt
from layers.PositionwiseFeedForward import PositionwiseFeedForward


"""
Transformer encoder layer mainly consists of two parts:
multi-headed self-attention and feed forward layer.
"""


class TransformencoderLayer(nn.Module):
    def __init__(self, args):
        super(TransformencoderLayer, self).__init__()

        self.self_attn = MultiHeadedAtt(args=args)
        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.feed_forward = PositionwiseFeedForward(args)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout(self.self_attn(hidden, hidden, hidden, mask))
        inter = self.layer_norm(inter + hidden)
        output = self.dropout(self.feed_forward(inter))
        output = self.layer_norm(output + inter)
        return output