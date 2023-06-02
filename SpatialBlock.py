import torch
import torch.nn as nn


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, fusion_types='channel_add'):
        super(SpatialBlock, self).__init__()
        # check if the fusion_type parameter is correct
        assert fusion_types in ['channel_add', 'channel_mul']
        self.fusion_types = fusion_types

        self.in_channels = in_channels

        # spatial attention / context modeling
        self.conv_pool = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

        # transform
        self.trans = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)

    def forward(self, x_gen, net):
        if self.fusion_types == 'channel_add':
            new_gen = net + x_gen
        else:
            new_gen = net * x_gen

        # spatial attention
        avgout = torch.mean(new_gen, dim=1, keepdim=True)
        maxout, _ = torch.max(new_gen, dim=1, keepdim=True)
        avg_max = torch.cat([avgout, maxout], dim=1)
        context_mask = self.conv_pool(avg_max)
        context_mask = self.sigmoid(context_mask)
        context = context_mask * new_gen

        # transform
        mask = self.trans(context)
        return mask

if __name__ == '__main__':
    model = SpatialBlock(16, fusion_types='channel_add')
    print(model)

    x = torch.randn(1, 16, 64, 64)
    out = model(x, x)
    print(out.shape)
