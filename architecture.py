import argparse
import sys

import torch

sys.path.append('vsrml')
from espnet.nets.pytorch_backend.transformer.encoder import Encoder as ConformerEncoder
from hparams import hparams
from utils import load_pretrained_resnet


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


def downsample_block(in_channels, out_channels, stride, group_norm=False):
    return torch.nn.Sequential(
        conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0
        ),
        torch.nn.BatchNorm2d(num_features=out_channels) if not group_norm else torch.nn.GroupNorm(32, num_channels=out_channels)
    )


def get_num_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


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

    def __init__(self, group_norm=False):
        super().__init__()

        self.conv_3d = conv3d(in_channels=1, out_channels=64, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                              padding=(2, 3, 3))
        self.batch_norm_3d = torch.nn.BatchNorm3d(num_features=64) if not group_norm else torch.nn.GroupNorm(32, num_channels=64)
        self.max_pool_3d = torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        x = self.conv_3d(x)
        x = self.batch_norm_3d(x)
        x = self.activation(x)
        x = self.max_pool_3d(x)  # same order as vsrml

        return x


class BasicBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion=1, down_sample=None, group_norm=False):
        super().__init__()

        self.conv_2d_1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=1)
        self.batch_norm_2d_1 = torch.nn.BatchNorm2d(num_features=out_channels) if not group_norm else torch.nn.GroupNorm(32, num_channels=out_channels)
        self.conv_2d_2 = conv2d(in_channels=in_channels * expansion, out_channels=out_channels, kernel_size=kernel_size,
                                stride=1, padding=1)
        self.batch_norm_2d_2 = torch.nn.BatchNorm2d(num_features=out_channels) if not group_norm else torch.nn.GroupNorm(32, num_channels=out_channels)
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
    """
    Based on https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/espnet/nets/pytorch_backend/backbones/conv3d_extractor.py
    """
    
    def __init__(self, group_norm=False):
        super().__init__()
        
        self.stem = Stem(group_norm=group_norm)
        self.res_block_1 = torch.nn.Sequential(
            BasicBlock(64, 64, 3, 1, group_norm=group_norm),
            BasicBlock(64, 64, 3, 1, group_norm=group_norm)
        )
        self.res_block_2 = torch.nn.Sequential(
            BasicBlock(64, 128, 3, 2, expansion=2, down_sample=downsample_block(64, 128, 2, group_norm=group_norm), group_norm=group_norm),
            BasicBlock(128, 128, 3, 1, group_norm=group_norm)
        )
        self.res_block_3 = torch.nn.Sequential(
            BasicBlock(128, 256, 3, 2, expansion=2, down_sample=downsample_block(128, 256, 2, group_norm=group_norm), group_norm=group_norm),
            BasicBlock(256, 256, 3, 1, group_norm=group_norm)
        )
        self.res_block_4 = torch.nn.Sequential(
            BasicBlock(256, 512, 3, 2, expansion=2, down_sample=downsample_block(256, 512, 2, group_norm=group_norm), group_norm=group_norm),
            BasicBlock(512, 512, 3, 1, group_norm=group_norm)
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


class ESPNetDecoder(torch.nn.Module): 
    
    def __init__(self, conformer_type='m', device='cpu'):
        super().__init__()

        self.conformer_params = hparams.conformer_params[conformer_type]    
        self.device = device
        self.num_input_features = 768
        
        # conformer contains initial linear layer before conformer layers i.e. fusion layer for resnet + speaker embedding features
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

    def __init__(self, conformer_type='m', device='cpu', pretrained_resnet_path=None, freeze_encoder=False, group_norm=False):
        super().__init__()

        # encoder can use batch or group-norm
        # decoder has a combination of layer (encoder layers) and batch-norm (conv module)
        self.encoder = Encoder(group_norm=group_norm)
        self.decoder = ESPNetDecoder(conformer_type=conformer_type, device=device)

        if pretrained_resnet_path: 
            self.encoder = load_pretrained_resnet(self.encoder, pretrained_resnet_path, freeze=freeze_encoder)

        num_encoder_params = get_num_parameters(self.encoder)
        num_decoder_params = get_num_parameters(self.decoder)
        total_params = num_encoder_params + num_decoder_params
        assert round(total_params / 1000000, 1) == hparams.conformer_params[conformer_type]['total_params']

        print('Encoder params:', num_encoder_params)
        print('Decoder params:', num_decoder_params)
        print('Total params:', total_params, '\n')

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

    net = V2S(conformer_type=args.conformer_size, device=device).to(device)
    batch_size = 6

    # testing same length input
    windows = torch.rand((batch_size, 1, hparams.fps, hparams.height, hparams.width)).to(device)
    lengths = torch.tensor([hparams.fps] * batch_size).to(device)
    speaker_embeddings = torch.rand((batch_size, 256, 1)).to(device)
    output = net(windows, speaker_embeddings, lengths)
    print('Same Length:', output.shape, '\n')

    # testing variable length input
    lengths = torch.tensor([50, 60, 70, 80, 90, 100])
    windows = [torch.rand((l, hparams.height, hparams.width)) for l in lengths]
    windows = torch.nn.utils.rnn.pad_sequence(windows, batch_first=True, padding_value=hparams.pad_value).unsqueeze(1)
    output = net(windows, speaker_embeddings[:len(lengths)], lengths)
    print('Variable Length:', output.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conformer_size', choices=['s', 'm', 'l'], default='s')

    main(parser.parse_args())
