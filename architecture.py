import torch
from torchaudio.models import Conformer
from torchinfo import summary

BATCH_SIZE = 32
TIMESTEPS = 20
HEIGHT = 88
WIDTH = 88
CONFORMER_PARAMS = {
    's': {
        'blocks': 6,
        'att_dim': 256,
        'att_heads': 4,
        'conv_k': 31,
        'ff_dim': 2048
    },
    'm': {
        'blocks': 12,
        'att_dim': 256,
        'att_heads': 4,
        'conv_k': 31,
        'ff_dim': 2048
    },
    'l': {
        'blocks': 12,
        'att_dim': 512,
        'att_heads': 8,
        'conv_k': 31,
        'ff_dim': 2048
    }
}


def conv3d(in_channels, out_channels, kernel_size, stride, padding):
    return torch.nn.Conv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )


def conv2d(in_channels, out_channels, kernel_size, stride, padding):
    return torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False
    )


class Stem(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv_3d = conv3d(in_channels=1, out_channels=64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                              padding=(2, 3, 3))
        self.batch_norm_3d = torch.nn.BatchNorm3d(num_features=64)
        self.max_pool_3d = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv_3d(x)
        x = self.batch_norm_3d(x)
        x = self.activation(x)
        x = self.max_pool_3d(x)

        return x


class BasicBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion=1, down_sample=None):
        super().__init__()

        self.conv_2d_1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=1)
        self.conv_2d_2 = conv2d(in_channels=in_channels * expansion, out_channels=out_channels, kernel_size=kernel_size,
                                stride=1, padding=1)
        self.down_sample = down_sample
        self.batch_norm_2d = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        residual = x  # skip connection
        x = self.conv_2d_1(x)
        x = self.batch_norm_2d(x)
        x = self.activation(x)
        x = self.conv_2d_2(x)
        x = self.batch_norm_2d(x)
        if self.down_sample:
            residual = self.down_sample(residual)
        x += residual
        x = self.activation(x)

        return x


class Encoder(torch.nn.Module):
    # https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/espnet/nets/pytorch_backend/backbones/conv3d_extractor.py

    def __init__(self):
        super().__init__()
        
        self.stem = Stem()
        self.res_block_1 = torch.nn.Sequential(
            BasicBlock(64, 64, 3, 1),
            BasicBlock(64, 64, 3, 1)
        )
        self.res_block_2 = torch.nn.Sequential(
            BasicBlock(64, 128, 3, 2, expansion=2, down_sample=conv2d(64, 128, 3, 2, 1)),
            BasicBlock(128, 128, 3, 1)
        )
        self.res_block_3 = torch.nn.Sequential(
            BasicBlock(128, 256, 3, 2, expansion=2, down_sample=conv2d(128, 256, 3, 2, 1)),
            BasicBlock(256, 256, 3, 1)
        )
        self.res_block_4 = torch.nn.Sequential(
            BasicBlock(256, 512, 3, 2, expansion=2, down_sample=conv2d(256, 512, 3, 2, 1)),
            BasicBlock(512, 512, 3, 1)
        )
        self.av_pool_2d = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.stem(x)
        x = x.reshape((BATCH_SIZE * TIMESTEPS, 64, 22, 22))
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.av_pool_2d(x)
        x = x.reshape((BATCH_SIZE, 512, TIMESTEPS))

        return x


class Decoder(torch.nn.Module):
    # https://github.com/hoangtuanvu/conformer_ocr uses similar Resnet + Conformer architecture

    def __init__(self, conformer_type='m', device='cpu'):
        super().__init__()

        self.device = device
        self.conformer_params = CONFORMER_PARAMS[conformer_type]

        self.conformer = Conformer(
            input_dim=self.conformer_params['att_dim'],
            num_heads=self.conformer_params['att_heads'],
            ffn_dim=self.conformer_params['ff_dim'],
            num_layers=self.conformer_params['blocks'],
            depthwise_conv_kernel_size=self.conformer_params['conv_k']
        )  # this is a conformer encoder w/ no positional encoder

        self.num_input_features = 768
        self.fc_1 = torch.nn.Linear(in_features=self.num_input_features,
                                    out_features=self.conformer_params['att_dim'])  # this is a linear layer
        self.fc_2 = torch.nn.Linear(in_features=self.conformer_params['att_dim'],
                                    out_features=320)  # this is a projection layer
        self.activation = torch.nn.ReLU()

        self.reset_parameters()

    def forward(self, x):
        # TODO: Can we pass variable length sequences during training?
        #  investigate if reshape is correct here
        lengths = torch.tensor(TIMESTEPS, dtype=torch.int32).repeat(BATCH_SIZE).to(self.device)  # input lengths, shape B. i-th in batch represents no. valid frames

        x = self.fc_1(x)
        x = self.activation(x)

        x, lengths = self.conformer(x, lengths)  # input = [B, T, N]

        x = self.fc_2(x)  # output = [B, T, 320]

        x = x.view(BATCH_SIZE, TIMESTEPS, 4, 80)

        return x.view(BATCH_SIZE, 80, 80)  # 80 frames per second

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.)

        for fc in [self.fc_1, self.fc_2]:
            init_weights(fc)

        self.conformer.apply(init_weights)


class V2S(torch.nn.Module):

    def __init__(self, conformer_type='m', device='cpu'):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(conformer_type=conformer_type, device=device)

    def forward(self, windows, speaker_embeddings):
        # 3D CNN + Resnet-18
        x = self.encoder(windows)  # input = [B, 1, T, H, W], output = [B, 512, T]

        # concat speaker embeddings
        # embeddings need repeated for T to concat i.e. same embedding per frame
        x = torch.cat((x, speaker_embeddings.repeat(1, 1, TIMESTEPS)), 1)  # output = [B, 768, T]
        x = x.permute(0, 2, 1)  # output = [B, T, 768]

        # conformer decoder
        x = self.decoder(x)

        return x


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using:', device)

    windows = torch.rand((BATCH_SIZE, 1, TIMESTEPS, HEIGHT, WIDTH)).to(device)
    speaker_embeddings = torch.rand((BATCH_SIZE, 256, 1)).to(device)

    net = V2S(conformer_type='s', device=device).to(device)
    print(summary(net, input_size=[(32, 1, 20, 88, 88), (32, 256, 1)], device=device))
    output = net(windows, speaker_embeddings)
    print(output.shape)


if __name__ == '__main__':
    main()
