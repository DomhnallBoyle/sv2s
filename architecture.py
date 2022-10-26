import argparse
import math
import sys

import torch
from torchaudio.models import Conformer
from torchinfo import summary

sys.path.append('vsrml')
from espnet.nets.pytorch_backend.transformer.encoder import Encoder as ConformerEncoder

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
DROPOUT_RATE = 0.2


def conv3d(in_channels, out_channels, kernel_size, stride, padding):
    # conv modules have a default weights init
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


def downsample_block(in_channels, out_channels, stride):
    return torch.nn.Sequential(
        conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0
        ),
        torch.nn.BatchNorm2d(num_features=out_channels)
    )


def get_num_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def init_weights(m):
    torch.nn.init.xavier_uniform_(m.weight)
    torch.nn.init.constant_(m.bias, 0.)


def initialise_weights(net, init_type):
    # https://github.com/espnet/espnet/blob/4138010fb66ad27a43e8bee48a4932829a0847ae/espnet/nets/pytorch_backend/transformer/initializer.py#L14 

    # weight and bias init
    for p in net.parameters():
        if p.dim() > 1:
            if init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(p.data)
            elif init_type == 'xavier_normal':
                torch.nn.init.xavier_normal_(p.data)
            elif init_type == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(p.data, nonlinearity='relu')
            else:
                raise ValueError('Unknown initialization:', init_type)
        elif p.dim() == 1:
            p.data.zero_()

    # reset some modules with default init
    for m in net.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm)):
            m.reset_parameters()



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
        self.batch_norm_2d_1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv_2d_2 = conv2d(in_channels=in_channels * expansion, out_channels=out_channels, kernel_size=kernel_size,
                                stride=1, padding=1)
        self.batch_norm_2d_2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.down_sample = down_sample
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        residual = x  # skip connection
        out = self.conv_2d_1(x)
        out = self.batch_norm_2d_1(out)
        out = self.activation(out)
        out = self.conv_2d_2(out)
        out = self.batch_norm_2d_2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.activation(out)

        return out


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
            BasicBlock(64, 128, 3, 2, expansion=2, down_sample=downsample_block(64, 128, 2)),
            BasicBlock(128, 128, 3, 1)
        )
        self.res_block_3 = torch.nn.Sequential(
            BasicBlock(128, 256, 3, 2, expansion=2, down_sample=downsample_block(128, 256, 2)),
            BasicBlock(256, 256, 3, 1)
        )
        self.res_block_4 = torch.nn.Sequential(
            BasicBlock(256, 512, 3, 2, expansion=2, down_sample=downsample_block(256, 512, 2)),
            BasicBlock(512, 512, 3, 1)
        )
        self.av_pool_2d = torch.nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        batch_size, timesteps = x.shape[0], x.shape[2]

        x = self.stem(x)
        x = x.view(batch_size * timesteps, 64, 22, 22)
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.av_pool_2d(x)
        x = x.view(batch_size, 512, timesteps)

        return x


class PositionalEncoder(torch.nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model, dropout_rate=0., max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        position = torch.arange(max_len).unsqueeze(1)  # uses no. timesteps
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # uses embedding size
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # from dim 0, steps of 2
        pe[:, 1::2] = torch.cos(position * div_term)  # from dim 1, steps of 2
        self.register_buffer('pe', pe)

    def forward(self, x):
        # # scaling: https://datascience.stackexchange.com/questions/87906
        # x = x * math.sqrt(self.d_model)

        # TODO: does it need scaled?

        x = x + self.pe[:x.shape[1], :]  # only uses based on the sequence lengths

        return self.dropout(x)


class Decoder(torch.nn.Module):
    # https://github.com/hoangtuanvu/conformer_ocr uses similar Resnet + Conformer architecture

    def __init__(self, conformer_type='m', device='cpu'):
        super().__init__()

        self.device = device
        self.conformer_params = CONFORMER_PARAMS[conformer_type]
        self.num_input_features = 768  # 512 encoder + 256 audio embedding features

        self.positional_encoder = PositionalEncoder(
            d_model=self.conformer_params['att_dim'],
        )

        self.conformer = Conformer(
            input_dim=self.conformer_params['att_dim'],
            num_heads=self.conformer_params['att_heads'],
            ffn_dim=self.conformer_params['ff_dim'],
            num_layers=self.conformer_params['blocks'],
            depthwise_conv_kernel_size=self.conformer_params['conv_k'],
            # dropout=DROPOUT_RATE  # defaulted to 0
        )  # this is a conformer encoder w/ no positional encoder

        self.fc_1 = torch.nn.Linear(in_features=self.num_input_features,
                                    out_features=self.conformer_params['att_dim'])  # this is a linear layer
        # self.dropout = torch.nn.Dropout(p=DROPOUT_RATE)
        self.activation = torch.nn.ReLU()

        self.fc_2 = torch.nn.Linear(in_features=self.conformer_params['att_dim'],
                                    out_features=320)  # this is a projection layer - no activation function

        self.reset_parameters()

    def forward(self, x, lengths):
        batch_size, timesteps = x.shape[:2]

        x = self.fc_1(x)
        x = self.activation(x)
        # x = self.dropout(x)

        x = self.positional_encoder(x)

        x, lengths = self.conformer(x, lengths)  # input = [B, T, N]

        x = self.fc_2(x)  # output = [B, T, 320]

        x = x.view(batch_size, timesteps, 4, 80)

        return x.view(batch_size, timesteps * 4, 80)  # 80 frames per second

    def reset_parameters(self):
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.)

        for fc in [self.fc_1, self.fc_2]:
            init_weights(fc)

        self.conformer.apply(init_weights)


class ESPNetDecoder(torch.nn.Module): 
    
    def __init__(self, conformer_type='m', device='cpu'):
        super().__init__()

        self.conformer_params = CONFORMER_PARAMS[conformer_type]    
        self.device = device
        self.num_input_features = 768
        
        # conformer contains initial linear layer before conformer layers
        # also contains positional encoding
        # RELU for ffd layer and swish for conv layer of conformer blocks
        self.conformer = ConformerEncoder(
            idim=self.num_input_features,
            attention_dim=self.conformer_params['att_dim'],
            attention_heads=self.conformer_params['att_heads'],
            linear_units=self.conformer_params['ff_dim'],
            num_blocks=self.conformer_params['blocks'],
            cnn_module_kernel=self.conformer_params['conv_k'],
            input_layer='linear',
            use_cnn_module=True,
            encoder_attn_layer_type='rel_mha',
            macaron_style=True
        )
        self.projection_layer = torch.nn.Linear(
            in_features=self.conformer_params['att_dim'],
            out_features=320
        )

        initialise_weights(self.conformer, init_type='xavier_uniform')
        initialise_weights(self.projection_layer, init_type='xavier_uniform')

    def forward(self, x, lengths): 
        batch_size, timesteps = x.shape[:2]

        # create masks from lengths
        max_length = lengths.max().int()
        masks = torch.arange(max_length).to(self.device).expand(batch_size, max_length) < lengths.unsqueeze(1)
        masks = masks.unsqueeze(1)  # requires [B, 1, T] https://github.com/espnet/espnet/issues/4567

        x, masks = self.conformer(x, masks)
        x = self.projection_layer(x)

        x = x.view(batch_size, timesteps, 4, 80)

        return x.view(batch_size, timesteps * 4, 80)
        

class V2S(torch.nn.Module):

    def __init__(self, conformer_type='m', device='cpu'):
        super().__init__()

        self.encoder = Encoder()
        # self.decoder = Decoder(conformer_type=conformer_type, device=device)
        self.decoder = ESPNetDecoder(conformer_type=conformer_type, device=device)

        num_encoder_params = get_num_parameters(self.encoder)
        num_decoder_params = get_num_parameters(self.decoder)
        print('Encoder params:', num_encoder_params)
        print('Decoder params:', num_decoder_params)
        print('Total params:', num_encoder_params + num_decoder_params, '\n')

    def forward(self, windows, speaker_embeddings, lengths):
        # 3D CNN + Resnet-18
        x = self.encoder(windows)  # input = [B, 1, T, H, W], output = [B, 512, T]

        # concat speaker embeddings
        # embeddings need repeated for T to concat i.e. same embedding per frame
        timesteps = windows.shape[2]
        x = torch.cat((x, speaker_embeddings.repeat(1, 1, timesteps)), 1)  # output = [B, 768, T]
        x = x.permute(0, 2, 1)  # output = [B, T, 768]

        # conformer decoder
        x = self.decoder(x, lengths)

        return x


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using:', device)

    windows = torch.rand((BATCH_SIZE, 1, TIMESTEPS, HEIGHT, WIDTH)).to(device)
    lengths = torch.tensor([TIMESTEPS] * BATCH_SIZE).to(device)
    speaker_embeddings = torch.rand((BATCH_SIZE, 256, 1)).to(device)

    net = V2S(conformer_type=args.conformer_size, device=device).to(device)

    # TODO: Fix this
    # print(summary(net, input_size=[(BATCH_SIZE, 1, TIMESTEPS, HEIGHT, WIDTH), (BATCH_SIZE, 256, 1), (BATCH_SIZE, 1)], device=device))

    output = net(windows, speaker_embeddings, lengths)
    print('Output:', output.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conformer_size', choices=['s', 'm', 'l'], default='s')

    main(parser.parse_args())
