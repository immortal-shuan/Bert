import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, args):
        super(PositionwiseFeedForward, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(args.hidden_size, args.feedforward_size),
            nn.GELU(),
            nn.Linear(args.feedforward_size, args.hidden_size)
        )

    def forward(self, x):
        output = self.layer(x)
        return output
