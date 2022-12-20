import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
from scipy import signal

from hparams import hparams
from utils import load_wav


def parallel_wavegan(mel_specs, model_checkpoint):
    if not model_checkpoint:
        raise Exception('Supply a model ckpt when using ParallelWaveGAN')
        
    tmp_directory = Path('/tmp/pwgan')  
    feats_path = tmp_directory.joinpath('feats.scp')
    if tmp_directory.exists():
        shutil.rmtree(tmp_directory)
    tmp_directory.mkdir()

    with feats_path.open('w') as f:
        for i, mel_spec in enumerate(mel_specs):
            mel_spec_path = tmp_directory.joinpath(f'mel_{i}.npy')
            np.save(mel_spec_path, mel_spec)
            f.write(f'dummy_{i} {str(mel_spec_path)}\n')

    command = f'parallel-wavegan-decode --checkpoint {model_checkpoint} --feats-scp {str(feats_path)} --outdir {str(tmp_directory)}'
    command = 'export CUDA_VISIBLE_DEVICES=\'\'; ' + command  # cpu usage
    subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return [load_wav(tmp_directory.joinpath(f'dummy_{i}_gen.wav')).astype(np.float64) 
            for i in range(len(mel_specs))]


def get_hop_size():
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)

    return hop_size


def _stft(y):
    return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(), win_length=hparams.win_size)


def _istft(y):
    return librosa.istft(y, hop_length=get_hop_size(), win_length=hparams.win_size)


def _griffin_lim(S):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)

    return y


def _denormalize(D):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_mels:
            return (((np.clip(D, -hparams.max_abs_value, hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

    if hparams.symmetric_mels:
        return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
    else:
        return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)


def _build_mel_basis():
    assert hparams.fmax <= hparams.sample_rate // 2

    return librosa.filters.mel(sr=hparams.sample_rate, n_fft=hparams.n_fft, n_mels=hparams.num_mels,
                               fmin=hparams.fmin, fmax=hparams.fmax)


def _mel_to_linear(mel_spectrogram):
    _inv_mel_basis = np.linalg.pinv(_build_mel_basis())

    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)

    return wav


def griffin_lim(mel_specs):
    """Converts mel spectrogram to waveform using librosa"""
    wavs = []
    for mel_spec in mel_specs:
        mel_spec = mel_spec.T

        if hparams.signal_normalization:
            D = _denormalize(mel_spec)
        else:
            D = mel_spec

        S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db))  # convert back to linear
        wavs.append(inv_preemphasis(_griffin_lim(S ** hparams.power), hparams.pre_emphasis, hparams.pre_emphasize))
    
    return wavs
