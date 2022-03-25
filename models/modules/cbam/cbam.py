import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channel_in: int, reduction_ratio: int):
        super(ChannelAttention, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in // reduction_ratio, out_features=channel_in)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # init
        avg_pool = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        max_pool = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # run
        avg_features = avg_pool(x)
        max_features = max_pool(x)
        channel_att_map_avg = self.shared_mlp(avg_features)
        channel_att_map_max = self.shared_mlp(max_features)
        stack_map = torch.stack(tensors=[channel_att_map_avg, channel_att_map_max], dim=0)
        summation_map = torch.sum(input=stack_map, dim=0)
        scaled = self.sigmoid(summation_map).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scaled


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(tensors=(torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)),
                         dim=1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1,
                      padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)
        x_spatial = self.spatial_attention(x_compress)
        scaled = self.sigmoid(x_spatial)
        return x * scaled


class CBAM(nn.Module):
    def __init__(self, channel_in, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def main():
    _input = torch.ones((7, 3, 112, 112))
    network = CBAM(channel_in=3)
    y = network(_input)
    print(y.shape)


if __name__ == '__main__':
    main()
