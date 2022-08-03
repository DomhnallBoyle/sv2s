import matplotlib.pyplot as plt
import numpy as np
import torch


def stft(x, fft_size, hop_size, win_length, window):
    # https://github.com/kan-bayashi/ParallelWaveGAN/blob/1f7949f593cc5600478cd4cc23bbf34bbcb0bcff/parallel_wavegan/losses/stft_loss.py#L16
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE: clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


def plot_spectrogram(pred_spectrogram, path, title=None, split_title=False, target_spectrogram=None, max_len=None, auto_aspect=False):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

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
    plt.savefig(path, format="png")
    plt.close()
