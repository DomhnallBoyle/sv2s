import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

FFMPEG_OPTIONS = '-hide_banner -loglevel error'
REPLACE_AUDIO_COMMAND = f'ffmpeg {FFMPEG_OPTIONS} -i {{input_video_path}} -i {{input_audio_path}} -map 0:v -map 1:a -c:v copy -shortest {{output_video_path}}'

log_path = None
plt.switch_backend('agg')


def log(s):
    print(s)
    with log_path.open('a') as f:
        f.write(f'{s}\n')


def plot_spectrogram(pred_spectrogram, title=None, target_spectrogram=None, max_len=None, auto_aspect=False, loss=None, video_path=None):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if loss is not None:
        title += f'\nLoss: {loss}'

    if video_path is not None:
        title += f'\nVideo Path: {video_path}'

    fig = plt.figure(figsize=(10, 8))
    # set common labels
    fig.text(0.5, 0.18, title, horizontalalignment="center", fontsize=16)

    # target spectrogram subplot
    if target_spectrogram is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)

        if auto_aspect:
            im = ax1.imshow(np.rot90(target_spectrogram), aspect="auto", interpolation="none")
        else:
            im = ax1.imshow(np.rot90(target_spectrogram), interpolation="none")
        ax1.set_title("Target Mel-Spectrogram")
        fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
        ax2.set_title("Predicted Mel-Spectrogram")
    else:
        ax2 = fig.add_subplot(211)

    if auto_aspect:
        im = ax2.imshow(np.rot90(pred_spectrogram), aspect="auto", interpolation="none")
    else:
        im = ax2.imshow(np.rot90(pred_spectrogram), interpolation="none")
    fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)

    plt.tight_layout()

    return fig


def norm_wav(wav):
    # between -1 and 1
    return (2 * (wav - min(wav)) / (max(wav) - min(wav))) - 1


def save_wav(wav, save_path, sr):
    # https://stackoverflow.com/a/10359645
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))

    wavfile.write(save_path, sr, wav.astype(np.int16))


def load_wav(path): 
    _, wav = wavfile.read(path)

    return wav


def overlay_audio(input_video_path, input_audio_path, output_video_path):
    # overlay audio onto video                             
    subprocess.run(REPLACE_AUDIO_COMMAND.format(
        input_video_path=input_video_path,
        input_audio_path=input_audio_path,
        output_video_path=output_video_path), 
    shell=True)


def load_pretrained_resnet(encoder, pretrained_path, freeze=False):
    pm = torch.load(pretrained_path)

    def copy(p, v):
        p.data.copy_(v)

    # stem
    stem = encoder.stem
    copy(stem.conv_3d.weight, pm['encoder.frontend.frontend3D.0.weight'])
    for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']: 
        p = getattr(stem.batch_norm_3d, attr)
        copy(p, pm[f'encoder.frontend.frontend3D.1.{attr}'])

    # trunks
    for i in range(1, 5):  # layer 
        for j in range(2):  # block
            # conv_2d_1
            copy(
                getattr(encoder, f'res_block_{i}')[j].conv_2d_1.weight,
                pm[f'encoder.frontend.trunk.layer{i}.{j}.conv1.weight']
            )

            # batch_norm_2d_1
            p = getattr(encoder, f'res_block_{i}')[j].batch_norm_2d_1
            for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                copy(
                    getattr(p, attr), 
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.bn1.{attr}']
                )

            # conv_2d_2
            copy(
                getattr(encoder, f'res_block_{i}')[j].conv_2d_2.weight,
                pm[f'encoder.frontend.trunk.layer{i}.{j}.conv2.weight']
            )

            # batch_norm_2d_1
            p = getattr(encoder, f'res_block_{i}')[j].batch_norm_2d_2
            for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                copy(
                    getattr(p, attr), 
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.bn2.{attr}']
                )

            # occurs in first block of layers 2-4
            is_downsample = i in [2, 3, 4] and j == 0
            if is_downsample: 
                copy(
                    getattr(encoder, f'res_block_{i}')[j].down_sample[0].weight,
                    pm[f'encoder.frontend.trunk.layer{i}.{j}.downsample.0.weight']
                )
                p = getattr(encoder, f'res_block_{i}')[j].down_sample[1]
                for attr in ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']:
                    copy(
                        getattr(p, attr),
                        pm[f'encoder.frontend.trunk.layer{i}.{j}.downsample.1.{attr}']
                    )

    # freeze everything except the batch norm layers
    # requires_grad = False = frozen = not updated
    for name, param in encoder.named_parameters(): 
        if freeze and 'batch_norm' not in name: 
            param.requires_grad = False 
        else:
            assert param.requires_grad

    total = 0
    for name, p in encoder.named_parameters():  # weights, biases only
        total += p.data.sum()
    assert total.int() == -45892, f'{total.int()} != -45892'  # taken from resnet

    return encoder
