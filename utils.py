import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

_inv_mel_basis = None
plt.switch_backend('agg')


def plot_spectrogram(pred_spectrogram, title=None, target_spectrogram=None, max_len=None, auto_aspect=False, loss=None):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if loss is not None:
        title += f'\nLoss: {loss}'

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


def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)

    return hop_size


def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size)


def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)


def _griffin_lim(S, hparams):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)

    return y


def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value, hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2

    return librosa.filters.mel(sr=hparams.sample_rate, n_fft=hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))

    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def spec_2_wav(mel_spec, hparams):
    """Converts mel spectrogram to waveform using librosa"""
    if hparams.signal_normalization:
        D = _denormalize(mel_spec, hparams)
    else:
        D = mel_spec

    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.pre_emphasis, hparams.pre_emphasize)


def save_wav(wav, save_path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))

    wavfile.write(save_path, sr, wav.astype(np.int16))


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
