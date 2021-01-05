import torch
import torch.nn as nn
from encoder.bert_encoder import BertEncoder
from layers.embeding import WordPosSegEmbedding


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = WordPosSegEmbedding(args)
        self.encoder = BertEncoder(args)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, args.labels_num)
        )

    def forward(self, input_ids, token_types, input_masks):

        """
        Args:
            input_ids: [batch_size x seq_length]
            token_types: [batch_size x seq_length]  
            input_mask: [batch_size x seq_length]
        """

        emb = self.embedding(input_ids, token_types)
        feature = self.encoder(emb, input_masks)
        avg_feature = self.avg_pool(feature.permute(0, 2, 1)).squeeze(-1)
        max_feature = self.max_pool(feature.permute(0, 2, 1)).squeeze(-1)
        
        output = self.fc(torch.cat((avg_feature, max_feature), dim=-1))
        return output
